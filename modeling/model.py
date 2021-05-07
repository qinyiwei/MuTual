from transformers import ElectraModel, ElectraPreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils

class ElectraForMultipleChoicePlus(ElectraPreTrainedModel):
    def __init__(self, config, add_GRU=True, bidirectional=False, word_level=True, add_cls = False):
        super().__init__(config)
        self.electra = ElectraModel(config)
        feature_dim = config.hidden_size
        if bidirectional:
            feature_dim += config.hidden_size
        if add_cls:
            feature_dim += config.hidden_size
        self.pooler = nn.Linear(feature_dim, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.add_GRU = add_GRU
        self.word_level = word_level
        self.add_cls = add_cls
        self.bidirectional = bidirectional
        if self.add_GRU:
            self.gru = nn.GRU(config.hidden_size,config.hidden_size,num_layers=1,batch_first = True, bidirectional=bidirectional)
        self.init_weights()
        print("add_GRU is: "+str(add_GRU))
        print("bidirectional is: "+str(bidirectional))
        print("word_level is:"+str(word_level))
        print("add_cls is:"+str(add_cls))

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos = None,
        position_ids=None,
        turn_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        sep_pos = sep_pos.view(-1, sep_pos.size(-1)) if sep_pos is not None else None
        turn_ids = turn_ids.view(-1, turn_ids.size(-1)) if turn_ids is not None else None
        
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        B = input_ids.shape[0]
        if self.add_GRU:
            if self.word_level:
                input_length = [(attention_mask[i]!=0).sum().item() for i in range(B)]
            else:
                input_length = [(sep_pos[i]!=0).sum().item() for i in range(B)]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        attention_mask = (1.0 - attention_mask) * -10000.0

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)
        cls_rep = sequence_output[:,0]
        if self.add_GRU:
            if self.word_level:
                sequence_output = torch.nn.utils.rnn.pack_padded_sequence(sequence_output, lengths=input_length,enforce_sorted = False,batch_first=True)
                sequence_output,last_state = self.gru(sequence_output)
                sequence_output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence_output,batch_first=True)
                if(self.bidirectional):
                    last_state = torch.cat([last_state[0],last_state[1]],dim=1)
                #index = torch.LongTensor(input_length).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,sequence_output.shape[-1]).to(self.device)
                #last_state = torch.gather(sequence_output, index=index-1, dim=1).squeeze(1)
                if(self.add_cls):
                    feature = torch.cat([cls_rep, last_state])
                else:
                    feature = last_state
            else:
                index = sep_pos.unsqueeze(-1).expand(-1,-1,sequence_output.shape[-1])
                sequence_output = torch.gather(sequence_output, index=index, dim=1)
                sequence_output = torch.nn.utils.rnn.pack_padded_sequence(sequence_output, lengths=input_length,enforce_sorted = False,batch_first=True)
                sequence_output,last_state = self.gru(sequence_output)
                sequence_output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence_output,batch_first=True)
                if(self.bidirectional):
                    last_state = torch.cat([last_state[0],last_state[1]],dim=1)
                #index = torch.LongTensor(input_length).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,sequence_output.shape[-1]).to(self.device)
                #last_state = torch.gather(sequence_output, index=index-1, dim=1).squeeze(1)
                if(self.add_cls):
                    feature = torch.cat([cls_rep, last_state],dim=1)
                else:
                    feature = last_state
        else:
            feature = cls_rep
        
        pooled_output = self.pooler_activation(self.pooler(feature))
        pooled_output = self.dropout(pooled_output)
        
        if num_labels > 2:
            logits = self.classifier(pooled_output)
        else:
            logits = self.classifier2(pooled_output)

        reshaped_logits = logits.view(-1, num_labels) if num_labels > 2 else logits

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs #(loss), reshaped_logits, (hidden_states), (attentions)

    @property
    def device(self):
        return self.pooler.weight.device

class NumericalReasoning(nn.Module):
    def __init__(self, hidden_size):
        self.linear_alpha = nn.Linear(hidden_size, 1)
        self.linear_f = nn.Linear(hidden_size, hidden_size)

        self.linears = []
        for i in range(8):
            self.linears.append(nn.Linear(hidden_size, hidden_size, bias=False))

    def forward(self, word_emb, num_ids, is_response, numbers):
        #word_emb:sequence_output [B*num_choice, seq_len, hidden_size]
        #num_ids:[B*num_choices,MAX_NUMBER]
        #is_response:[B*num_choices, MAX_NUMBER] 0:number from utterances, 1:number from response, -1:padding
        #numbers:[B*num_choices, MAX_NUMBER]
        InitEmbedding = word_emb[:,num_ids] #[B*num_choices, MAX_NUMBER, hidden_size]
        #node relatedness Measure
        alpha = F.sigmoid(self.linear_alpha(InitEmbedding)) #[B*num_choices, MAX_NUMBER, 1]
        #Message Propagation
        TempEmbedding = torch.zeros_like(word_emb) #[B*num_choices, MAX_NUMBER, hidden_size]

        B = TempEmbedding.shape[0] #B*num_choices

        for b in range(B):
            num_nodes = (num_ids[b]!=-1).sum()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if(i!=j):
                        r = choose(is_response[b],numbers[b],i,j)
                        TempEmbedding[b,i] += alpha[b,j]*self.linears[r](InitEmbedding[b,j])
                TempEmbedding[b,i] /= (num_nodes-1)


        #Node Representation Update
        NumbEmbedding = F.relu(self.linear_f(InitEmbedding)+TempEmbedding)#[B*num_choices, MAX_NUMBER, hidden_size]

        return NumbEmbedding #[B*num_choices, MAX_NUMBER, hidden_size]

    def choose(is_response, numbers, i,j):
        r = 0
        r += 4*(1 if numbers[i]>numbers[j] else 0)
        r += 2*(1 if is_response[i]==1 else 0)
        r += 1*(1 if is_response[j]==1 else 0)
        return r