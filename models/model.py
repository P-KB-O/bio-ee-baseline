import torch.nn as nn
from torch.nn.modules import GRU, Dropout, Embedding, Linear


def self_attention(input):
    pass


class BiGruAtten(nn.Module):
    def __init__(self, word2vecf, entity_embedding, word2vecf_dim=200, entity_dim=50, hidden_dim=200, outputdim=50):
        super(BiGruAtten, self).__init__()

        self.word2vecf_layer = Embedding.from_pretrained(word2vecf)
        self.entity_embedding = Embedding.from_pretrained(entity_embedding)

        self.input_dim = word2vecf_dim + entity_dim
        self.hidden_dim = self.atten_dim = hidden_dim

        self.GRU_layer = GRU(input_size=self.input_dim, hidden_dim=hidden_dim, dropout=0.3, bidirectional=True)
        self.Drop_layer = Dropout(p=0.5)

        self.output_layer = Linear(in_features=self.hidden_dim+self.atten_dim, out_features=outputdim)

    def forward(self, x):
        pass


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
