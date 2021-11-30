import random

import torch
from torch import nn
from functools import partial
from RecBole.recbole.model.abstract_recommender import SequentialRecommender
from RecBole.recbole.model.layers import TransformerEncoder
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class Masking(nn.Module):
    def __init__(self):
        super(Masking, self).__init__()
    def forward(self,x,item_seq):
        mask_index = (item_seq == 0).nonzero()
        b_axis = mask_index[:,0]
        s_axis = mask_index[:,1]
        x[b_axis,s_axis] = torch.zeros(x[b_axis,s_axis].size(),device=item_seq.device)
        return x

class MLPMixerB(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MLPMixerB, self).__init__(config, dataset)
        cuda0 = torch.device('cuda:0')
        self.item_feat = dataset.get_item_feature()
        self.class_feat = self.item_feat.interaction['class'].to(cuda0)
        self.class_feat = torch.cat((self.class_feat,torch.unsqueeze(self.class_feat[0],0)),0)
        self.n_layers = config['n_layers']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        
        self.mask_ratio = config['mask_ratio']
        
        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']
        
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        
        expansion_factor = 4
        chan_first = partial(nn.Conv1d, kernel_size = 1)
        chan_last = nn.Linear
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.class_embedding = nn.Linear(6,self.hidden_size)
        self.masking = Masking()
        self.tokenMixer = PreNormResidual(self.hidden_size*2, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.channelMixer = PreNormResidual(self.hidden_size*2, FeedForward(self.hidden_size*2, expansion_factor, self.hidden_dropout_prob, chan_last))
        self.LayerNorm = nn.LayerNorm(self.hidden_size*2,eps=self.layer_norm_eps)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence
    
    def reconstruct_train_data(self, item_seq):
        """
        Mask item sequence for training.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []
        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            neg_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if item == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    neg_item.append(self._neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, neg_items, masked_index

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position - 1] = self.mask_token
        return item_seq               
                    
        
    def forward(self, item_seq):
        class_emb = self.class_embedding(self.class_feat[item_seq].float())
        item_emb = self.item_embedding(item_seq)
        mixer_output = torch.cat((item_emb,class_emb),2)
        # mixer_output = self.masking(input_emb,item_seq)
        for _ in range(self.n_layers):
            mixer_output = self.tokenMixer(mixer_output)
            mixer_output = self.channelMixer(mixer_output)
        mixer_output = self.LayerNorm(mixer_output)
        return mixer_output  # [B L H]
    
    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.
        Examples:
            sequence: [1 2 3 4 5]
            masked_sequence: [1 mask 3 mask 5]
            masked_index: [1, 3]
            max_length: 5
            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        masked_item_seq, pos_items, neg_items, masked_index = self.reconstruct_train_data(item_seq)
        seq_output = self.forward(masked_item_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        if self.loss_type == 'BPR':
            pos_items_emb = self.item_embedding(pos_items)  # [B mask_len H]
            neg_items_emb = self.item_embedding(neg_items)  # [B mask_len H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B mask_len]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B mask_len]
            targets = (masked_index > 0).float()
            loss = - torch.sum(torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets) \
                   / torch.sum(targets)
            return loss

        elif self.loss_type == 'CE':
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            test_item_emb = self.item_embedding.weight[:self.n_items]  # [item_num H]
            test_class_emb = self.class_embedding(self.class_feat[list(range(self.n_items))].float())
            test_item_emb = torch.cat((test_item_emb,test_class_emb),1)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
            targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

            loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
                   / torch.sum(targets)
            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
            
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq[:,1:])
        seq_output = self.gather_indexes(seq_output, item_seq_len - 2)  # [B H]
        # dim(seq_output) = (batch size, embedding dimension)
        test_item_emb = self.item_embedding(test_item)
        test_class_emb = self.class_embedding(self.class_feat[test_item].float())
        # dim(test_item_emb) = (batch size, embedding dimension)
        test_item_emb = torch.cat((test_item_emb,test_class_emb),1)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        # dim(scores) = batch size
        return scores
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq[:,1:])
        seq_output = self.gather_indexes(seq_output, item_seq_len - 2)  # [B H]
        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores
