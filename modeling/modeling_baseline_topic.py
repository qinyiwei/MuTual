from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Conv1d
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils

BertLayerNorm = torch.nn.LayerNorm

ACT2FN = {"gelu": F.gelu, "relu": F.relu}


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_state, num_heads=1):
        super().__init__()

        self.q_linear = nn.Linear(hidden_state, hidden_state)
        self.v_linear = nn.Linear(hidden_state, hidden_state)
        self.k_linear = nn.Linear(hidden_state, hidden_state)
        self.attention = nn.MultiheadAttention(hidden_state, num_heads)

    def forward(self, query_input, input, mask=None):
        query = self.q_linear(query_input)
        key = self.k_linear(input)
        value = self.v_linear(input)

        attn_output, attn_output_weights = self.attention(query, key, value, mask)

        return attn_output

class GRUWithPadding(nn.Module):
    def __init__(self, hidden_size, num_rnn = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_rnn
        self.biGRU = nn.GRU(hidden_size, hidden_size, self.num_layers, batch_first = True, bidirectional = False)

    def forward(self, inputs):
        batch_size = len(inputs)
        sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1].size(0), reverse = True)
        idx_inputs = [i[0] for i in sorted_inputs]
        inputs = [i[1] for i in sorted_inputs]
        inputs_lengths = [len(i[1]) for i in sorted_inputs]

        inputs = rnn_utils.pad_sequence(inputs, batch_first = True)
        inputs = rnn_utils.pack_padded_sequence(inputs, inputs_lengths, batch_first = True) #(batch_size, seq_len, hidden_size)

        h0 = torch.rand(self.num_layers, batch_size, self.hidden_size).to(inputs.data.device) # (2, batch_size, hidden_size)
        self.biGRU.flatten_parameters()
        out, _ = self.biGRU(inputs, h0) # (batch_size, 2, hidden_size )
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first = True) # (batch_size, seq_len, 2 * hidden_size)

        _, idx2 = torch.sort(torch.tensor(idx_inputs))
        idx2 = idx2.to(out_pad.device)
        output = torch.index_select(out_pad, 0, idx2)
        out_len = out_len.to(out_pad.device)
        out_len = torch.index_select(out_len, 0, idx2)

        #out_idx = (out_len - 1).unsqueeze(1).unsqueeze(2).repeat([1,1,self.hidden_size*2])
        out_idx = (out_len - 1).unsqueeze(1).unsqueeze(2).repeat([1,1,self.hidden_size])
        output = torch.gather(output, 1, out_idx).squeeze(1)

        return output

class ElectraForMultipleChoicePlus(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.electra = ElectraModel(config)
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 2)

        self.gru = GRUWithPadding(config.hidden_size * 2)
        self.attention = MultiHeadAttention(config.hidden_size)
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
        topic_ids = None,
        topic_mask = None
    ):
        topic_ids = topic_ids.view(-1, topic_ids.size(-1)) if topic_ids is not None else None
        topic_mask = topic_mask.view(-1, topic_mask.size(-1)) if topic_mask is not None else None

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
        #===== gather topic words =====
        #topic_ids (batch_size * num_choice,80)
        topic_ids = topic_ids.unsqueeze(2).repeat([1,1,self.hidden_size])
        topic_sequence = torch.gather(sequence_output,1, topic_ids).permute(1, 0, 2)# batch, seq, hidden
        sequence_output = sequence_output.permute(1,0,2)
        topic_output = self.attention(sequence_output, topic_sequence, (1-topic_mask).bool()) # batch, seq, hidden
        #topic_output = torch.max(topic_output, 0)[0].unsqueeze(1)
        sequence_output = torch.cat((sequence_output, topic_output),dim=2).permute(1,0,2)

        context_utterance_level = []
        batch_size = sequence_output.size(0)
        utter_size = sep_pos.shape[1]
        #topic_output = topic_output.repeat([1, utter_size, 1])  # batch, utter, hidden
        topic_length = topic_output.shape[1]
        for batch_idx in range(batch_size):
            context_utterances = [torch.max(sequence_output[batch_idx, :(sep_pos[batch_idx][0] + 1)], dim=0, keepdim=True)[0]]

            for j in range(1, utter_size):
                if sep_pos[batch_idx][j] == 0:

                    break
                current_context_utter, _ = torch.max(sequence_output[batch_idx, (sep_pos[batch_idx][j - 1] + 1):(sep_pos[batch_idx][j] + 1)],
                                                     dim=0, keepdim=True)

                context_utterances.append(current_context_utter)

            single_topic = topic_output[batch_idx].repeat([j, 1])
            context_utterance_level.append(
                torch.cat(context_utterances, dim=0))  # (batch_size, utterances, hidden_size)

        gru_output = self.gru(context_utterance_level)


        pooled_output = self.pooler_activation(self.pooler(gru_output)) #(batch_size * num_choice, seq_len, hidden_size)
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

