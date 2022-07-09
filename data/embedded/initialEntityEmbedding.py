import numpy as np
import pickle


clazz = ['a', 'b', 'c']
embedding_dim=50

clazz_num = len(clazz)
with open('entity_embedding.pkl', 'wb') as fw:
    pickle.dump(np.random.normal(size=(clazz_num, embedding_dim)), fw)

with open('entity_vocab.pkl', 'wb') as fw:
    pickle.dump(clazz, fw)

