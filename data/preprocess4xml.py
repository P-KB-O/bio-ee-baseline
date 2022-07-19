try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import pickle
import os
import argparse


class PreProcessor(object):

    def __init__(self, output_dir, data_name):
        self.output_dir = output_dir
        self.data_name = data_name

        # count data
        self.trigger_class_ids = {}
        self.count_trigger_class = 0

        self.entity_class_ids = {}
        self.count_entity_class = 0

        self.dep_class_ids = {}
        self.count_dep_class = 0

        self.special_char = ["-", "[", "(", ")", "]", "/"]

        self.non_find = []
        rf = open(os.path.join(self.output_dir, "non_find"), 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line[0:len(line) - 1]
            self.non_find.append(line)
        rf.close()

    def process(self, file_, train=True):
        assert os.path.splitext(file_)[-1] == '.xml', 'not xml format'

        if train:
            output_file = os.path.join(self.output_dir, self.data_name, 'train_')
        else:
            output_file = os.path.join(self.output_dir, self.data_name, 'test_')

        wf1 = open(output_file + "token.txt", 'w', encoding='utf-8')  # golden sentences
        wf2 = open(output_file + "dep.txt", 'w', encoding='utf-8')  # dependency parse
        wf3 = open(output_file + "label.txt", 'w', encoding='utf-8')  # trigger label
        wf4 = open(output_file + "entity_type.txt", 'w', encoding='utf-8')  # entity type
        wf5 = open(output_file + "interaction.txt", 'w', encoding='utf-8')
        wf6 = open(output_file + "offset_id.txt", 'w', encoding='utf-8')
        wf7 = open(output_file + 'duplicated.txt', 'w', encoding='utf-8')

        tree = ET.parse(file_)
        root = tree.getroot()

        max_len = 0
        wrong = 0
        duplicated_tri = {}  # aim at getting attention of triggers with different ids
        duplicated_tri_ids = {}
        count_duplicated = 0

        sen_doc_ids = []  # the doc id of each sentence.

        for document in root:
            doc_id = document.get("origId")
            for sentence in document:
                sen_doc_ids.append(doc_id[:])
                line = ""   # used for appending tokens.
                entities = sentence.findall("entity")

                # ============ tri info===============
                tri_ids = []
                tri_offsets = []
                tri_types = []
                tri_texts = []

                all_train_triggers = []

                # =========== entity info============
                entity_ids = []
                entity_offsets = []
                entity_types = []
                entity_texts = []

                # entity information
                if entities is not None:
                    for entity in entities:
                        if entity.get("origBANNEROffset") is not None:
                            continue

                        char_offset = entity.get("charOffset")
                        eid = entity.get("id")
                        e_type = entity.get("type")
                        text = entity.get("text")
                        head_offset = entity.get("headOffset")

                        # deal with lack of headOffset in some dataset
                        # added by eleve11 in 2022.07.19
                        if head_offset is None:
                            texts = text.split(' ')
                            if len(texts) == 1:
                                head_offset = char_offset
                            else:
                                h = str(int(char_offset.split('-')[-1]) - len(texts[-1]))  # head
                                t = char_offset.split('-')[-1]  # tail
                                head_offset = h + '-' + t

                        # print(char_offset, eid, e_type, head_offset, text)

                        s1 = int(char_offset.split("-")[0])
                        s2 = int(head_offset.split("-")[0])
                        e1 = int(char_offset.split("-")[1])
                        e2 = int(head_offset.split("-")[1])

                        if head_offset != char_offset and s1 >= s2 and e1 <= e2:
                            char_offset = head_offset

                        if entity.get('given') is not None:
                            # example entity
                            # <entity charOffset="0-5" given="True" headOffset="0-5" id="GE09.d150.s3.e4"
                            # origId="10022882.T5" origOffset="419-424" text="5-LOX" type="Protein" />
                            entity_ids.append(eid)
                            entity_offsets.append(char_offset)
                            entity_types.append(e_type)
                            entity_texts.append(text)
                        else:
                            if char_offset not in tri_offsets:
                                if len(text.split(" ")) > 1:    # check whether there are triggers containing multiple words.
                                    wrong += 1
                                # example event
                                # <entity charOffset="46-57" event="True" headOffset="46-57" id="GE09.d150.s3.e23"
                                # negation="True" origId="10022882.T24" origOffset="465-476" text="coexpressed"
                                # type="Gene_expression" />
                                tri_ids.append(eid)
                                tri_offsets.append(char_offset)
                                tri_types.append(e_type)
                                tri_texts.append(text)
                                if train:
                                    all_train_triggers.append(tri_texts)
                            else:
                                index0 = tri_offsets.index(char_offset)
                                duplicated_tri_ids[eid] = tri_ids[index0]
                                if char_offset in duplicated_tri.keys():
                                    duplicated_tri[char_offset] += "#" + eid
                                else:
                                    duplicated_tri[char_offset] = eid
                                count_duplicated += 1

                # event-arguments interaction
                interactions = sentence.findall("interaction")
                interaction_e1 = ""
                interaction_e2 = ""
                interaction_type = ""
                if interactions is not None:
                    for interaction in interactions:
                        event = interaction.get("event")
                        if event != 'True':
                            continue
                        e1 = interaction.get("e1")
                        e2 = interaction.get("e2")
                        inter_type = interaction.get("type")
                        interaction_e1 += str(e1) + " "
                        interaction_e2 += str(e2) + " "
                        interaction_type += str(inter_type) + " "

                pos = ""  # pos of sentence
                char_offsets = ""
                count = 0

                # ============ token info===============
                for token in sentence.find("analyses").find("tokenization"):
                    text = token.get("text")
                    char_offset = token.get("charOffset")
                    start = int(char_offset.split("-")[0])
                    end = int(char_offset.split("-")[1])

                    tok_pos = token.get("POS")  # part of speech of token
                    if len(text) > 1:
                        for sc in self.special_char:
                            text = text.strip(sc)

                    if train and text in self.non_find and '-' in text:
                        if text in tri_texts:
                            texts = text.split("-")
                            for i, word in enumerate(texts):
                                line += word + " "
                                pos += tok_pos + " "

                                if i == len(texts) - 1:
                                    char_offsets += str(start) + "-" + str(end) + " "
                                else:
                                    char_offsets += str(start) + "-" + str(start + len(word)) + " "
                                    start = start + len(word)
                                count += 1
                        else:
                            line += text + " "
                            char_offsets += char_offset + " "
                            pos += tok_pos + " "
                            count += 1
                    elif not train and text in self.non_find and '-' in text:
                        if text in all_train_triggers:
                            texts = text.split("-")
                            for i, word in enumerate(texts):
                                line += word + " "
                                pos += tok_pos + " "
                                # note that I get rid of '-'.
                                if i == len(texts) - 1:
                                    char_offsets += str(start) + "-" + str(end) + " "
                                else:
                                    char_offsets += str(start) + "-" + str(start + len(word)) + " "
                                    start = start + len(word)
                                count += 1
                        else:
                            line += text + " "
                            char_offsets += char_offset + " "
                            pos += tok_pos + " "
                            count += 1
                    else:
                        line += text + " "
                        char_offsets += char_offset + " "
                        pos += tok_pos + " "
                        count += 1

                # ============ dep info===============
                dep_info = ""
                for dependency in sentence.find("analyses").find("parse"):
                    if dependency.tag == "dependency":
                        # print(dependency.get("id"))
                        dep_info += dependency.get("t1")[3:] + "#"
                        dep_info += dependency.get("t2")[3:] + "#"
                        dep_type = dependency.get("type")
                        dep_info += dep_type + " "

                line = line.strip()
                char_offsets = char_offsets.strip()
                pos = pos.strip()

                wf1.write(line + "\n")

                if count > max_len:
                    max_len = count

                tri_offsets, tri_types, tri_ids = self.sort_offset(tri_offsets, tri_types, tri_ids)
                entity_offsets, entity_types, entity_ids = self.sort_offset(entity_offsets, entity_types, entity_ids)
                label_temp, self.count_trigger_class = self.get_label_type(char_offsets, tri_offsets, tri_types,
                                                                           self.count_trigger_class,
                                                                           self.trigger_class_ids)
                label_temp = label_temp.strip()
                wf3.write(label_temp + "\n")

                label_temp, self.count_entity_class = self.get_label_type(char_offsets, entity_offsets, entity_types,
                                                                          self.count_entity_class,
                                                                          self.entity_class_ids)
                label_temp = label_temp.strip()
                wf4.write(label_temp + "\n")
                wf2.write(dep_info + "\n")

                wf5.write(interaction_e1.strip() + "#" + interaction_e2.strip() +
                          "#" + interaction_type.strip() + "\n")
                wf6.write(char_offsets + "#" + self.list2str(tri_offsets) + "#" +
                          self.list2str(tri_ids) + "#" + self.list2str(entity_offsets) +
                          "#" + self.list2str(entity_ids) + "\n")

        for key, value in duplicated_tri.items():
            wf7.write(key + "*" + value + "\n")
        wf7.close()

        wf1.close()
        wf2.close()
        wf3.close()
        wf4.close()
        wf5.close()
        wf6.close()

        print("longest sentence has " + str(max_len) + " words.")  # 125

        wf = open(output_file + "sen_doc_ids.pk", 'wb')
        pickle.dump(sen_doc_ids, wf)
        wf.close()

    def sort_offset(self, tri_offsets=[], tri_types=[], tri_ids=[]):
        temp = []
        for i in range(len(tri_offsets)):
            # print(tri_offsets[i])
            temp.append(int(tri_offsets[i].split("-")[0]))
        num = len(temp)

        for i in range(num - 1):
            for j in range(num - i - 1):
                if temp[j] > temp[j + 1]:
                    temp[j], temp[j + 1] = temp[j + 1], temp[j]
                    tri_offsets[j], tri_offsets[j+1] = tri_offsets[j+1], tri_offsets[j]
                    tri_types[j], tri_types[j+1] = tri_types[j+1], tri_types[j]
                    tri_ids[j], tri_ids[j+1] = tri_ids[j+1], tri_ids[j]

        return tri_offsets, tri_types, tri_ids

    def list2str(self, info_list=[]):
        res = ""
        for i in range(len(info_list)):
            res += str(info_list[i]) + " "
        return res.strip()

    def get_label_type(self, char_offsets, offsets, types, num, class_ids):
        char_offsets = char_offsets.split()
        label = ""
        j = 0
        signal = False

        if len(offsets) == 0:
            for i in range(len(char_offsets)):
                if "O" not in class_ids:
                    class_ids["O"] = num
                    num += 1
                label += "O "
            return label, num

        for i in range(len(char_offsets)):
            if j < len(offsets):
                s1 = int(char_offsets[i].split("-")[0])
                e1 = int(char_offsets[i].split("-")[1])
                s2 = int(offsets[j].split("-")[0])
                e2 = int(offsets[j].split("-")[1])
                if s1 >= s2 and e1 <= e2:
                    if signal and e1 == e2:
                        label_type = "E-" + types[j]
                        j += 1
                        signal = False
                    elif signal:
                        label_type = "I-" + types[j]
                    elif e1 == e2 and s1 == s2:
                        label_type = "S-" + types[j]
                        j += 1
                    else:
                        label_type = "B-" + types[j]
                        signal = True
                    label += " " + label_type
                    if label_type not in class_ids:
                        # print(label_type)
                        class_ids[label_type] = num
                        num += 1
                else:
                    label += " O"
                    if "O" not in class_ids:
                        class_ids["O"] = num
                        num += 1
            else:
                label += " O"
                if "O" not in class_ids:
                    class_ids["O"] = num
                    num += 1
        return label, num

    def write_ids(self):

        wf = open(os.path.join(self.output_dir, self.data_name, "tri_ids.txt"), 'w', encoding='utf-8')
        for key, value in self.trigger_class_ids.items():
            wf.write(key + "\t" + str(value) + "\n")
        wf.close()

        wf = open(os.path.join(self.output_dir, self.data_name, "entity_ids.txt"), 'w', encoding='utf-8')
        for key, value in self.entity_class_ids.items():
            wf.write(key + "\t" + str(value) + "\n")
        wf.close()

    def __str__(self):
        return '<{}> dataset: {} trigger classes, {} entity classes, {} dep classes'.format(self.data_name,
                                                                              self.count_trigger_class,
                                                                              self.count_entity_class,
                                                                              self.count_dep_class)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process xml format dataset.')
    parser.add_argument('--data', type=str, default='GE09', help='dataset options: GE09, GE11, BB11')
    parser.add_argument('--input_dir', type=str, default='./xml', help='input dir')
    args = parser.parse_args()

    data_name = args.data
    input_dir = args.input_dir

    train_file = os.path.join(input_dir, data_name + '-train.xml')
    test_file = os.path.join(input_dir, data_name + '-test.xml')

    output_based_dir = './preprocessed'
    output_dir = os.path.join(output_based_dir, data_name)
    if os.path.exists(output_dir):
        p = PreProcessor(output_based_dir, data_name)
    else:
        os.mkdir(output_dir)
        p = PreProcessor(output_based_dir, data_name)

    p.process(train_file, train=True)
    p.process(test_file, train=True)

    p.write_ids()
    print(p)
