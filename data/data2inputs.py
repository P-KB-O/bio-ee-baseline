import os
import pickle
import argparse

# https://stackoverflow.com/questions/57767854/keras-preprocessing-text-tokenizer-equivalent-in-pytorch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor


class Data2Inputs(object):
    def __init__(self, dir_, data_name, max_len):
        self.dir_ = dir_
        self.data_name = data_name
        self.max_len = max_len

    def get_data(self, train=True):
        inputs, labels, entity_labels, deps = self.read_instance_files(train)
        return inputs, labels, entity_labels, deps

    def read_instance_files(self, train=True):
        """
        read preprocessed data from files.
        here we read token, entity type and dependency clues.
        """
        if train:
            output_file = os.path.join(self.dir_, self.data_name, "train_")
        else:
            output_file = os.path.join(self.dir_, self.data_name, "test_")

        rf1 = open(output_file + "token.txt", 'r', encoding='utf-8')
        rf2 = open(output_file + "label.txt", 'r', encoding='utf-8')
        rf3 = open(output_file + "entity_type.txt", 'r', encoding='utf-8')
        rf4 = open(output_file + "dep.txt", 'r', encoding='utf-8')

        inputs = []
        labels = []
        entity_labels = []
        deps = []

        while True:
            line = rf1.readline()
            if line == "":
                break
            inputs.append(line)
            line = rf2.readline()
            labels.append(line)
            line = rf3.readline()
            entity_labels.append(line)
            line = rf4.readline()
            deps.append(line)

        rf1.close()
        rf2.close()
        rf3.close()
        rf4.close()

        return inputs, labels, entity_labels, deps

    def convert_text_to_index(self, train_inputs, test_inputs, padding=False):
        print("start convert text to index.")
        encoder = StaticTokenizerEncoder(train_inputs, tokenize=lambda s: s.split())
        train_inputs = [encoder.encode(example) for example in train_inputs]
        test_inputs = [encoder.encode(example) for example in test_inputs]
        print("finish!")
        self.write_ids(encoder.token_to_index, "word_index.pk")
        if padding:
            self.write_word_inputs(self.pad_inputs(train_inputs))
            self.write_word_inputs(self.pad_inputs(test_inputs), False)
        else:
            self.write_word_inputs(train_inputs)
            self.write_word_inputs(test_inputs, False)

    def pad_inputs(self, inputs):
        return stack_and_pad_tensors([pad_tensor(input[:self.max_len], self.max_len) for input in inputs])

    def write_ids(self, ids={}, file=""):
        wf = open(os.path.join(self.dir_, self.data_name, file), 'wb')
        pickle.dump(ids, wf)
        wf.close()

    def write_word_inputs(self, inputs=[], train=True):
        if train:
            output_file = os.path.join(self.dir_, self.data_name, "train_input.txt")
        else:
            output_file = os.path.join(self.dir_, self.data_name, "test_input.txt")

        wf = open(output_file, 'w', encoding='utf-8')
        for sentence in inputs:
            for index in sentence:
                wf.write(str(index) + " ")
            wf.write("\n")
        wf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process xml format dataset.')
    parser.add_argument('--dest_dir', type=str, default='./preprocessed')
    parser.add_argument('--data', type=str, default='GE09', help='dataset options: GE09, GE11, BB11')
    parser.add_argument('--seq_len', type=int, default=125)
    parser.add_argument('--padding', action="store_true", default=125)

    args = parser.parse_args()
    dir_ = args.dest_dir
    data_name = args.data
    max_len = args.seq_len
    padding = args.padding

    d = Data2Inputs(dir_, data_name, max_len)

    train_inputs, _, _, _ = d.get_data()
    test_inputs, _, _, _ = d.get_data(train=False)

    d.convert_text_to_index(train_inputs=train_inputs, test_inputs=test_inputs, padding=padding)
