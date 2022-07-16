import os
import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models import model, util

# ************* hyper-parameters start *************
max_epochs = 30
batchsize = 32
max_len = 100
# ************* hyper-parameters ended *************

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()

dataset_ = util.MultiDataset()
dataloader = DataLoader(dataset_, shuffle=True, batch_size=batchsize, collate_fn=dataset_.collate_fn)

word2vecf = util.load_word2vecf()
entity_embedding = util.load_entity_embedding()
model_ = model.BiGruAtten(word2vecf, entity_embedding)

optimizer = optim.Adam(model_.parameters(), lr=3e-4)

model_.train()
start = time.time()
for epoch in range(max_epochs):
    total_loss = 0
    print("====== epoch " + str(epoch + 1) + " ======")
    for _ in dataloader:
        avg_loss = 0

    print("{}: epoch {}, loss={}".format(time.time() - start, epoch + 1, avg_loss))


