import os
import argparse

import numpy as np
import pickle


def random_init_entity(path_, entity_classes, embedding_dim=50):
    embedding = np.random.normal(size=(entity_classes+2, embedding_dim))
    row, col = embedding.shape

    with open(os.path.join(path_, 'entity_type_matrix.txt'), 'w+') as fw:
        for i in range(row):
            line = ' '.join(list(map(str, embedding[i])))
            # print(line)
            fw.write(line)
            fw.write('\n')


class EmbeddingReader(object):
    """
    construct embedding matrix and pickle them.
    """

    def __init__(self, embedding_dir='./embedded'):
        self.embedding_dir = embedding_dir

        self.WORD_EMBEDDING_DIM = 200
        self.ENTITY_EMBEDDING_DIM = 50

        self.word_embedding_file = os.path.join(self.embedding_dir, "dim200vecs")

    def trim_word_embedding(self, word_num=20000, word_index={}, output_dir=''):
        rf = open(self.word_embedding_file, 'r', encoding='utf-8')

        embeddings_index = {}

        for line in rf:
            values = line.split()
            index = len(values) - self.WORD_EMBEDDING_DIM
            if len(values) > (self.WORD_EMBEDDING_DIM + 1):
                word = ""
                for i in range(len(values) - self.WORD_EMBEDDING_DIM):
                    word += values[i] + " "
                word = word.strip()
            else:
                word = values[0]
            # print(line)
            coefs = np.asarray(values[index:], dtype='float32')
            embeddings_index[word] = coefs

        rf.close()

        num_words = min(word_num, len(word_index))
        print("word num: " + str(num_words))
        embedding_matrix = np.zeros((num_words + 2, self.WORD_EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= word_num:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        wf = open(os.path.join(output_dir, "embedding_matrix.pk"), 'wb')
        pickle.dump(embedding_matrix, wf)
        wf.close()

    def read_ids(self, file):
        rf = open(file, 'rb')
        return pickle.load(rf)

    def load_embedding_file(self, file):
        rf = open(file, 'r', encoding='utf-8')
        entity_type_matrix = []
        while True:
            line = rf.readline()
            if line == "":
                break
            temp = line.strip().split()
            for i in range(len(temp)):
                temp[i] = float(temp[i])
            entity_type_matrix.append(temp)
        rf.close()
        entity_type_matrix = np.array(entity_type_matrix)
        wf = open(os.path.splitext(file)[0] + ".pk", 'wb')
        pickle.dump(entity_type_matrix, wf)
        wf.close()
        return entity_type_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process xml format dataset.')
    parser.add_argument('--dest_dir', type=str, default='./preprocessed')
    parser.add_argument('--data', type=str, default='GE09', help='dataset options: GE09, GE11, BB11')
    parser.add_argument('--seq_len', type=int, default=125)
    parser.add_argument('--entity_classes', type=int, default=6)

    args = parser.parse_args()
    dir_ = args.dest_dir
    data_name = args.data
    entity_classes = args.entity_classes
    max_len = args.seq_len

    random_init_entity(os.path.join(dir_, data_name), entity_classes=entity_classes)

    e = EmbeddingReader()
    word_index = e.read_ids(os.path.join(dir_, data_name, "word_index.pk"))
    # print(word_index)

    e.trim_word_embedding(word_index=word_index, output_dir=os.path.join(dir_, data_name))

    entity_embedding_file = "entity_type_matrix.txt"
    e.load_embedding_file(os.path.join(dir_, data_name, entity_embedding_file))
