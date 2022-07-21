import os
import pickle

from torch.utils.data import Dataset


def load_pickle(file):
    rf = open(file, 'rb')
    embedding_matrix = pickle.load(rf)
    rf.close()
    return embedding_matrix


class MultiDataset(Dataset):
    def __init__(self, dir_, data_name, train=True):
        super(MultiDataset, self).__init__()
        self.word_inputs, self.entity_inputs, self.labels, self.attention_labels \
            = self.load_data(dir_, data_name, train)
        # print(self.word_inputs.shape, self.entity_inputs.shape, self.labels.shape, self.attention_labels.shape)

    def load_data(self, dir_, data_name, train=True):
        if train:
            path = os.path.join(dir_, data_name, "train_")
        else:
            path = os.path.join(dir_, data_name, "test_")

        rf = open(path + "input.pk", 'rb')
        word_inputs = pickle.load(rf)
        rf.close()

        rf = open(path + "entity_inputs.pk", 'rb')
        entity_inputs = pickle.load(rf)
        rf.close()

        rf = open(path + "labels.pk", 'rb')
        labels = pickle.load(rf)
        rf.close()

        rf = open(path + "attention_label.pk", 'rb')
        attention_labels = pickle.load(rf)
        rf.close()

        return word_inputs, entity_inputs, labels, attention_labels

    def __len__(self):
        return len(self.word_inputs)

    def __getitem__(self, idx):
        return [self.word_inputs[idx], self.entity_inputs[idx], self.labels[idx], self.attention_labels[idx]]

    def collate_fn(self, batch):
        pass
