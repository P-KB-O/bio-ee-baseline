import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import GRU, Dropout, Embedding, Linear


def soft_attention_alignment(input_):
    attention = torch.matmul(input_, input_.transpose(-2, -1))
    w_att = F.softmax(attention, dim=-1)
    in_aligned = torch.matmul(w_att, input_)
    return in_aligned, w_att


class BiGruAtten(nn.Module):
    def __init__(self, word2vecf, entity_embedding, word2vecf_dim=200, entity_dim=50, hidden_dim=200, outputdim=50, freezing=True):
        super(BiGruAtten, self).__init__()

        self.word2vecf_layer = Embedding.from_pretrained(torch.tensor(word2vecf).to(torch.float32), freeze=freezing)
        self.entity_embedding = Embedding.from_pretrained(torch.tensor(entity_embedding).to(torch.float32), freeze=False)

        self.input_dim = word2vecf_dim + entity_dim
        self.hidden_dim = self.atten_dim = hidden_dim

        self.input_layer = Linear(in_features=word2vecf_dim, out_features=word2vecf_dim)
        self.GRU_layer = GRU(input_size=self.input_dim, hidden_size=hidden_dim, dropout=0.3, bidirectional=True)
        self.Drop_layer = Dropout(p=0.5)

        self.output_layer = Linear(in_features=self.hidden_dim*2+self.atten_dim, out_features=outputdim)

    def forward(self, sen_input, entity_type_input):
        sentence_embedding = self.word2vecf_layer(sen_input)
        entity_embedding = self.entity_embedding(entity_type_input)

        x = self.input_layer(sentence_embedding)
        self_attention_embedding, self_attention = soft_attention_alignment(x)

        inputs = torch.cat((sentence_embedding, entity_embedding), -1)
        encoded_sentence_embedding, _ = self.GRU_layer(inputs)

        x = torch.cat((encoded_sentence_embedding, self_attention_embedding), -1)
        x = self.Drop_layer(x)

        pred = self.output_layer(x)

        return pred, self_attention


# deprecated
# due to linear layer can calculate multi-dim tensor directly
class TimeDistributed(nn.Module):
    # pytorch version TimeDistributed
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


# https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440/14
class MyCriterion(nn.Module):
    def __init__(self):
        super(MyCriterion, self).__init__()
        # self.criterion1 = nn.MultiLabelMarginLoss()
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()

    def forward(self, pred, pred_attention, label, label_attention):
        # https://stackoverflow.com/questions/62213536/get-the-cross-entropy-loss-in-pytorch-as-in-keras

        return self.criterion1(torch.tensor(pred).transpose(-2, -1), torch.tensor(label).argmax(dim=-1)) + self.criterion2(pred_attention, label_attention)


