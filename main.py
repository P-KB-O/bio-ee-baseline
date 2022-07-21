import os
import time
import warnings

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models import model, util, load_data

warnings.filterwarnings('ignore')

# ************* basic-statics start *************
num_words = 11208
trigger_class = 40
entity_type_num = 5
embedding_dim = 200
entity_dim = 50
model_dir = './ckpt'  # checkpoints
dir_ = './data/preprocessed'
data_name = 'GE09'
is_train = True
# ************* basic-statics ended *************

# ************* hyper-parameters start *************
max_epochs = 30
batchsize = 64
max_len = 125
# ************* hyper-parameters ended *************

word2vecf = load_data.load_pickle(os.path.join(dir_, data_name, 'embedding_matrix.pk'))
entity_embedding = load_data.load_pickle(os.path.join(dir_, data_name, 'entity_type_matrix.pk'))
index_ids = load_data.load_pickle(os.path.join(dir_, data_name, 'tri_index_ids.pk'))


criterion = model.MyCriterion()

dataset_ = load_data.MultiDataset(dir_, data_name, train=is_train)
dataloader = DataLoader(dataset_, shuffle=True, batch_size=batchsize)

model_ = model.BiGruAtten(word2vecf, entity_embedding, outputdim=trigger_class)

optimizer = optim.Adam(model_.parameters(), lr=3e-4)

model_.train()
start = time.time()
for epoch in range(max_epochs):
    total_loss = 0
    print("====== epoch " + str(epoch + 1) + " ======")
    counter = 0
    trigs_len = 0
    guess_len = 0
    for sen_input, entity_type_input, train_labels, train_attention_labels in dataloader:
        pred, self_attention = model_(sen_input, entity_type_input)
        optimizer.zero_grad()

        loss = criterion(pred, self_attention, train_labels, train_attention_labels)

        loss.backward()
        optimizer.step()

        counter += 1
        total_loss += loss.item()

        t = torch.tensor(train_labels).argmax(dim=-1)
        tmp = torch.nonzero(torch.tensor(train_labels).argmax(dim=-1))
        trigs_len += len(torch.nonzero(torch.tensor(train_labels).argmax(dim=-1)))
        for el in tmp:
            if pred[el[0]][el[1]][t[el[0]][el[1]]].item() > 0.4:
                guess_len += 1

    avg_loss = total_loss / counter
    print("{}: epoch {}, loss={}, acc={}".format(time.ctime(), epoch + 1, avg_loss, guess_len / trigs_len))


