from transformers import ElectraModel, ElectraPreTrainedModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils

class GRUWithPadding(nn.Module):
    def __init__(self, config, num_rnn = 1, bidirectional=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_rnn
        self.biGRU = nn.GRU(config.hidden_size, config.hidden_size, self.num_layers, batch_first = True, bidirectional = bidirectional) 
        self.bidirectional = bidirectional

    def forward(self, inputs):
        batch_size = len(inputs)
        sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1].size(0), reverse = True)
        idx_inputs = [i[0] for i in sorted_inputs]
        inputs = [i[1] for i in sorted_inputs]
        inputs_lengths = [len(i[1]) for i in sorted_inputs]

        inputs = rnn_utils.pad_sequence(inputs, batch_first = True)
        inputs = rnn_utils.pack_padded_sequence(inputs, inputs_lengths, batch_first = True) #(batch_size, seq_len, hidden_size)

        h0 = torch.rand(self.num_layers*(2 if self.bidirectional else 1), batch_size, self.hidden_size).to(inputs.data.device) # (2, batch_size, hidden_size)
        self.biGRU.flatten_parameters()
        out, _ = self.biGRU(inputs, h0) # (batch_size, 2, hidden_size )
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first = True) # (batch_size, seq_len, 2 * hidden_size)

        _, idx2 = torch.sort(torch.tensor(idx_inputs))
        idx2 = idx2.to(out_pad.device)
        output = torch.index_select(out_pad, 0, idx2)
        out_len = out_len.to(out_pad.device)
        out_len = torch.index_select(out_len, 0, idx2)

        if self.bidirectional:
            out_idx = (out_len - 1).unsqueeze(1).unsqueeze(2).repeat([1,1,self.hidden_size*2])
        else:
            out_idx = (out_len - 1).unsqueeze(1).unsqueeze(2).repeat([1,1,self.hidden_size])
        output = torch.gather(output, 1, out_idx).squeeze(1)

        return output
    
class ElectraForMultipleChoice(ElectraPreTrainedModel):
    def __init__(self, config, add_GRU= True, bidirectional=True, add_cls = True):
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

        self.add_cls = add_cls
        self.bidirectional = bidirectional
        self.add_GRU = add_GRU
            
        if self.add_GRU:
            self.gru = GRUWithPadding(config,bidirectional=bidirectional)


        self.init_weights()
        print("add_GRU is: "+str(add_GRU))
        print("bidirectional is: "+str(bidirectional))
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
            context_utterance_level = []
            batch_size = sequence_output.size(0)
            utter_size = sep_pos.shape[1]
            for batch_idx in range(batch_size):
                context_utterances = [torch.max(sequence_output[batch_idx, :(sep_pos[batch_idx][0] + 1)], dim=0, keepdim=True)[0]]

                for j in range(1, utter_size):
                    if sep_pos[batch_idx][j] == 0:
                        #context_utterances.append(topic_output[batch_idx].unsqueeze(0))
                        break
                    current_context_utter, _ = torch.max(sequence_output[batch_idx, (sep_pos[batch_idx][j - 1] + 1):(sep_pos[batch_idx][j] + 1)],
                                                        dim=0, keepdim=True)

                    context_utterances.append(current_context_utter)


                context_utterance_level.append(
                    torch.cat(context_utterances, dim=0))  # (batch_size, utterances, hidden_size)

            feature = self.gru(context_utterance_level)
        else:
            feature = sequence_output

        if(self.add_cls):
            feature = torch.cat([cls_rep, feature],dim=1)

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

class ElectraForMultipleChoiceOther(ElectraPreTrainedModel):
    def __init__(self, config, add_GRU=True, bidirectional=False, word_level=False, add_cls = False, word_and_sent=False):
        super().__init__(config)
        self.electra = ElectraModel(config)
        feature_dim = config.hidden_size
        if bidirectional:
            feature_dim += config.hidden_size
        if word_and_sent:
            feature_dim *= 2
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
        self.word_and_sent = word_and_sent
        if self.add_GRU:
            self.gru = GRUWithPadding(config)
            #self.gru = nn.GRU(config.hidden_size,config.hidden_size,num_layers=1,batch_first = True, bidirectional=bidirectional)
        if self.word_and_sent:
            self.gru2 = nn.GRU(config.hidden_size,config.hidden_size,num_layers=1,batch_first = True, bidirectional=bidirectional)
        self.init_weights()
        print("add_GRU is: "+str(add_GRU))
        print("bidirectional is: "+str(bidirectional))
        print("word_level is:"+str(word_level))
        print("add_cls is:"+str(add_cls))
        print("word_and_sent is:"+str(word_and_sent))

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
            if self.word_and_sent:
                    sequence_output_sent = sequence_output
                    sequence_output = torch.nn.utils.rnn.pack_padded_sequence(sequence_output, lengths=input_length,enforce_sorted = False,batch_first=True)
                    sequence_output,last_state_word = self.gru(sequence_output)
                    sequence_output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence_output,batch_first=True)
                    if(self.bidirectional):
                        last_state_word = torch.cat([last_state_word[0],last_state_word[1]],dim=1)
                    else:
                        last_state_word = last_state_word.squeeze(0)

                    index = sep_pos.unsqueeze(-1).expand(-1,-1,sequence_output_sent.shape[-1])
                    sequence_output_sent = torch.gather(sequence_output_sent, index=index, dim=1)
                    sequence_output_sent = torch.nn.utils.rnn.pack_padded_sequence(sequence_output_sent, lengths=input_length,enforce_sorted = False,batch_first=True)
                    sequence_output_sent,last_state_sent = self.gru2(sequence_output_sent)
                    sequence_output_sent, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence_output_sent,batch_first=True)
                    if(self.bidirectional):
                        last_state_sent = torch.cat([last_state_sent[0],last_state_sent[1]],dim=1)
                    else:
                        last_state_sent = last_state_sent.squeeze(0)
                    last_state = torch.cat([last_state_word,last_state_sent],dim=1)
                    if(self.add_cls):
                        feature = torch.cat([cls_rep, last_state],dim=1)
                    else:
                        feature = last_state
            else:
                if self.word_level:#word_level GRU
                    sequence_output = torch.nn.utils.rnn.pack_padded_sequence(sequence_output, lengths=input_length,enforce_sorted = False,batch_first=True)
                    sequence_output,last_state = self.gru(sequence_output)
                    sequence_output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence_output,batch_first=True)
                    if(self.bidirectional):
                        last_state = torch.cat([last_state[0],last_state[1]],dim=1)
                    else:
                        last_state = last_state.squeeze(0)
                    if(self.add_cls):
                        feature = torch.cat([cls_rep, last_state],dim=1)
                    else:
                        feature = last_state
                else:#sent_level GRU
                    context_utterance_level = []
                    batch_size = sequence_output.size(0)
                    utter_size = sep_pos.shape[1]
                    for batch_idx in range(batch_size):
                        context_utterances = [torch.max(sequence_output[batch_idx, :(sep_pos[batch_idx][0] + 1)], dim=0, keepdim=True)[0]]
                        for j in range(1, utter_size):
                            if sep_pos[batch_idx][j] == 0:
                                #context_utterances.append(topic_output[batch_idx].unsqueeze(0))
                                break
                            current_context_utter, _ = torch.max(sequence_output[batch_idx, (sep_pos[batch_idx][j - 1] + 1):(sep_pos[batch_idx][j] + 1)],
                                                                dim=0, keepdim=True)

                            context_utterances.append(current_context_utter)
                        context_utterance_level.append(
                            torch.cat(context_utterances, dim=0))  # (batch_size, utterances, hidden_size)
                    gru_output = self.gru(context_utterance_level)
                    feature = gru_output
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

class ElectraForMultipleChoicePlus(ElectraPreTrainedModel):
    def __init__(self, config, num_rnn = 1,bidirectional=True,add_cls=False):
        super().__init__(config)

        self.electra = ElectraModel(config)

        self.SASelfMHA = MHA(config)
        self.SACrossMHA = MHA(config)

        #self.fuse2 = FuseLayer(config)
        
        self.gru1 = GRUWithPadding(config, num_rnn, bidirectional)
        self.gru2 = GRUWithPadding(config, num_rnn, bidirectional)
        self.gru3 = GRUWithPadding(config, num_rnn, bidirectional)

        feature_dim = config.hidden_size * 3
        if bidirectional:
            feature_dim *= 2
        if add_cls:
            feature_dim += config.hidden_size
        self.pooler = nn.Linear(feature_dim , config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 2)

        self.bidirectional = bidirectional
        self.add_cls = add_cls
        
        self.init_weights()

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
        
        #print("sep_pos:", sep_pos)
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        #print("size of sequence_output:", sequence_output.size())
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        
        # (batch_size * num_choice, 1, 1, seq_len)

        local_mask = torch.zeros_like(attention_mask, dtype = self.dtype)
        local_mask = local_mask.repeat((1,1,attention_mask.size(-1), 1)) #(batch_size * num_choice, 1, seq_len, seq_len)
        global_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_self_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_cross_mask = torch.zeros_like(local_mask, dtype = self.dtype)

        last_seps = []

        for i in range(input_ids.size(0)):
            last_sep = 1

            while last_sep < len(sep_pos[i]) and sep_pos[i][last_sep] != 0: 
                last_sep += 1
            
            last_sep = last_sep - 1
            last_seps.append(last_sep)

            local_mask[i, 0, turn_ids[i] == turn_ids[i].T] = 1.0
            local_mask[i, 0, :, (sep_pos[i][last_sep] + 1):] = 0

            sa_self_mask[i, 0, (turn_ids[i] % 2) == (turn_ids[i].T % 2)] = 1.0
            sa_self_mask[i, 0, :, (sep_pos[i][last_sep] + 1):] = 0

            global_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)] = 1.0 - local_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)]
            sa_cross_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)] = 1.0 - sa_self_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)]

        attention_mask = (1.0 - attention_mask) * -10000.0
        local_mask = (1.0 - local_mask) * -10000.0
        global_mask = (1.0 - global_mask) * -10000.0
        sa_self_mask = (1.0 - sa_self_mask) * -10000.0
        sa_cross_mask = (1.0 - sa_cross_mask) * -10000.0

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
        
        sa_self_word_level = self.SASelfMHA(sequence_output, sequence_output, attention_mask = sa_self_mask)[0]
        sa_cross_word_level = self.SACrossMHA(sequence_output, sequence_output, attention_mask = sa_cross_mask)[0]


        context_word_level = sequence_output
        sa_word_level = sa_self_word_level
        sa_word_level2 = sa_cross_word_level

        new_batch = []

        context_utterance_level = []
        sa_utterance_level = []
        sa_utterance_level2 = []

        for i in range(sequence_output.size(0)):
            context_utterances = [torch.max(context_word_level[i, :(sep_pos[i][0] + 1)], dim = 0, keepdim = True)[0]]
            sa_utterances = [torch.max(sa_word_level[i, :(sep_pos[i][0] + 1)], dim = 0, keepdim = True)[0]]
            sa_utterances2 = [torch.max(sa_word_level2[i, :(sep_pos[i][0] + 1)], dim = 0, keepdim = True)[0]]

            for j in range(1, last_seps[i] + 1):
                current_context_utter, _ = torch.max(context_word_level[i, (sep_pos[i][j-1] + 1):(sep_pos[i][j] + 1)], dim = 0, keepdim = True)
                current_sa_utter, _ = torch.max(sa_word_level[i, (sep_pos[i][j-1] + 1):(sep_pos[i][j] + 1)], dim = 0, keepdim = True)
                current_sa_utter2, _ = torch.max(sa_word_level2[i, (sep_pos[i][j-1] + 1):(sep_pos[i][j] + 1)], dim = 0, keepdim = True)
                context_utterances.append(current_context_utter)
                sa_utterances.append(current_sa_utter)
                sa_utterances2.append(current_sa_utter2)

            context_utterance_level.append(torch.cat(context_utterances, dim = 0)) # (batch_size, utterances, hidden_size)
            sa_utterance_level.append(torch.cat(sa_utterances, dim = 0))
            sa_utterance_level2.append(torch.cat(sa_utterances2, dim = 0))

        context_final_states = self.gru1(context_utterance_level) 
        sa_final_states = self.gru2(sa_utterance_level) # (batch_size * num_choice, 2 * hidden_size)
        sa_final_states2 = self.gru3(sa_utterance_level2) # (batch_size * num_choice, 2 * hidden_size)

        final_state = torch.cat((context_final_states, sa_final_states, sa_final_states2), 1)
        if(self.add_cls):
            final_state = torch.cat([cls_rep, final_state],dim=1)

        pooled_output = self.pooler_activation(self.pooler(final_state))
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

class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim = -1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim = -1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim = -1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        
        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

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