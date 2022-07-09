# copy and rewrite word2vecf/scripts/vec2nps.py
# https://github.com/BIU-NLP/word2vecf/blob/master/scripts/vecs2nps.py
import sys

import numpy as np
import pickle


fh = open(sys.argv[1])
foutname = sys.argv[2]
first = fh.readline()
size = list(map(int, first.strip().split()))
wvecs = np.zeros((size[0],size[1]),float)

vocab=[]
for i,line in enumerate(fh):
    line = line.strip().split()
    vocab.append(line[0])
    wvecs[i,] = np.array(list(map(float, line[1:])))

np.save(foutname + ".npy", wvecs)   # to numpy format
with open(foutname + '.pkl', 'wb') as file:  # to pickle format
    pickle.dump(wvecs, file)

with open(foutname+".vocab","w") as outf:
   print(" ".join(vocab), file=outf)  # write into outf


with open(foutname + '_vocab.pkl', 'wb') as file: # to pickle format
    pickle.dump(vocab, file)
