import os
import pickle


def load_data(dir_, data_name, train=True):
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


def load_pickle(file):
    rf = open(file, 'rb')
    embedding_matrix = pickle.load(rf)
    rf.close()
    return embedding_matrix