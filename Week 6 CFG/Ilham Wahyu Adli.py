#!/usr/bin/env python
# coding: utf-8

# Import library yang dibutuhkan

# In[1]:


import nltk
from nltk.parse.generate import generate
from nltk.parse import ViterbiParser
from nltk.corpus import BracketParseCorpusReader
from nltk import Nonterminal, nonterminals, Production, PCFG, induce_pcfg
import time
from functools import reduce
import sys


# In[2]:


grammarCFG = nltk.CFG.fromstring("""
  S -> NP VP 
  NP -> Det N | RPR 
  VP -> V | V NP | V PP | VP PP | V N | N PP | V PNP
  PNP -> N PP 
  PP -> IN N 
  V -> "memberi" | "melihat" | "memfoto"
  RPR -> "Saya" 
  Det -> "para" | "sang" | "si"
  N -> "tetangga" | "tahu" | "beruang" | "kacamata" |  "piring" |  "Ilham"
  IN ->  "ke" | "dengan" | "bersama"
  """)
print(grammarCFG.is_flexible_chomsky_normal_form())


# In[3]:


# kalimatTest = 'Saya melihat beruang dengan kacamata'
kalimatTest = 'Saya memberi tahu dengan piring ke tetangga'
kalimatTest = 'Saya memfoto beruang bersama Ilham'
kalimatTest = kalimatTest.split()


# In[4]:


td_parser = nltk.parse.TopDownChartParser(grammarCFG)

for tree in td_parser.parse(kalimatTest):
    print(tree)


# In[5]:


bu_parser = nltk.parse.BottomUpChartParser(grammarCFG)

for tree in bu_parser.parse(kalimatTest):
    print(tree)


# # BAGIAN2

# In[6]:

ptb = BracketParseCorpusReader(r"", r".*/*\.mrg")
#load Tree Bank


# Induksi PCFG (Probabilistic Context Free Grammar) dari constituency Treebank

# In[7]:


S = Nonterminal('S')

productions = []
for t in ptb.parsed_sents():
    productions += t.productions()
grammarTB = induce_pcfg(S, productions)
# print(grammarTB)


# In[8]:


kalimatTest = 'monyet mengganggu warga Delhi'
kalimatTest = 'pemerintah amankan monyet berekor panjang'
kalimatTest = kalimatTest.split()


# In[9]:


# contoh menggunakan bottom-up parser
bu_parser = nltk.parse.TopDownChartParser(grammarTB)
for hasil in bu_parser.parse(kalimatTest):
    print(hasil)


# In[10]:


tokens = kalimatTest
parser = ViterbiParser(grammarTB)
all_parses = {}
parser.trace(3)
t = time.time()
parses = parser.parse_all(tokens)
time = time.time() - t
average = (
    reduce(lambda a, b: a + b.prob(), parses, 0) / len(parses) if parses else 0
)
num_parses = len(parses)
for p in parses:
    all_parses[p.freeze()] = 1

print("Hasil")

print()

for parse in parses:
    print(parse)


# In[ ]:




