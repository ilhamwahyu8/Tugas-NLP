#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import gensim.models
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


from gensim.test.utils import datapath
from gensim import utils

class MyCorpus(object): 
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        path = 'dataset'
        files = os.listdir(path)
#         corpus_path = datapath('dataset/') # sesuaikan dengan path masing-masing
        for file in files: 
            for line in open(path + "/" + file, 'r', encoding='utf-8'):
                # assume there's one document per line, tokens separated by whitespace
                # asumsi 1 dokumen adalah 1 kalimat, dituliskan per baris. Antar token dipisahkan dengan spasi
                yield utils.simple_preprocess(line)


# In[3]:


sentences = MyCorpus()
w2v_model = gensim.models.Word2Vec(sentences=sentences, size=200, iter=80, min_count=1, workers = 8)


# In[4]:


print(w2v_model.wv.similarity('staycation', 'liburan'))
print(w2v_model.wv.similarity('cirebon', 'hotel'))
print(w2v_model.wv.similarity('cirebon', 'kepala'))


# In[5]:


from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


# In[6]:


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


# In[7]:


x_vals, y_vals, labels = reduce_dimensions(w2v_model)


# In[8]:


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 20)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))


# In[10]:


plot_with_matplotlib(x_vals, y_vals, labels)


# In[ ]:




