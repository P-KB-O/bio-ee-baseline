# bio-ee-baseline

here is a repo about biomedical event extraction. here is a **pytorch** version of [DUT-LiuYang/biomedical-event-trigger-extraction](https://github.com/DUT-LiuYang/biomedical-event-trigger-extraction).

`Paper:` Li L, Liu Y. [Exploiting argument information to improve biomedical event trigger identification via recurrent neural networks and supervised attention mechanisms](https://ieeexplore.ieee.org/document/8217711)[C]//2017 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2017: 565-568.

## biomedical datasets

you can use [TEES](https://github.com/jbjorne/TEES) to get BioNLP[09-13] datasets which format are xml.

**surprise!** I found a fantastic biomedical github repo [bigscience-workshop/biomedical](https://github.com/bigscience-workshop/biomedical)

> BigBIO (BigScience Biomedical) is an open library of biomedical dataloaders built using Huggingface's (🤗)

## Prerequisites
First, I get PudMed abstracts data by crawling [PudMed E-utilizes api](https://www.ncbi.nlm.nih.gov/books/NBK25497/) 👍 . [here](https://github.com/P-KB-O/bio-misc/blob/main/PubMed/getPubMedAbstracts_multithread.py) is my spider code.

In paper, author use **GDep parser** for dependency parses, but I can't get this software. In this repo, so we use stanford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) for dependency parses and apply its result to train dependency-base word embedding.(word2vecf code from [BIU-NLP/word2vecf](https://github.com/BIU-NLP/word2vecf), a kind of a variant of word2vec)

## Usage

preprocess stage:

``` shell
cd data
./preprocess.sh
```


train stage:

``` python
python main.py
```

test stage:

``` python
python test.py
```