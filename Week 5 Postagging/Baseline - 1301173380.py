import pandas as pd
import nltk
from nltk import word_tokenize
import os
import copy
nltk.download('punkt')

path = 'Dataset'
trainSet = pd.read_csv(path + "/Train Set.tsv",header = None, delimiter="\t")
testSet = pd.read_csv(path + "/Test Set.tsv",header = None, delimiter="\t")

kalimatTrain = []
insertKalimat = ''
for i in range(len(trainSet)):
    insertKalimat += str(trainSet.loc[i,0]).lower()
    insertKalimat += '/'
    insertKalimat += str(trainSet.loc[i,1])
    insertKalimat += '@'
    if trainSet.loc[i,0] == "." and trainSet.loc[i,1] == "Z":
        kalimatTrain.append(insertKalimat)
        insertKalimat = ''
# Pembuatan 50 kalimat dari kata di dalam trainset

word_tag_freq = {} # dictionary untuk menyimpan frekuensi tag kata
tag_count = {} # dictionary untuk menyimpan frekuensi tag
for i in range(0, len(kalimatTrain)):
    tokens = kalimatTrain[i].split('@')
    tokens.pop()
    for j in range(0, len(tokens)):
        pair = tokens[j].split('/')
        word = pair[0].lower()
        tag = pair[1]

        # simpan di dictionary tag
        if tag in tag_count:
            tag_count[tag] = tag_count[tag] + 1
        else:
            tag_count[tag] = 1
            
        # simpan di dictionary kata,tag
        key = (word,tag)
        if key in word_tag_freq:
            word_tag_freq[key] = word_tag_freq[key] + 1
        else:
            word_tag_freq[key] = 1

def get_most_freq_tag():
    most_freq_tag = max(tag_count, key=tag_count.get) 
    return most_freq_tag
default_tag = get_most_freq_tag()

def get_most_freq_tag_word(word):
    word_tags = {k: v for k, v in word_tag_freq.items() if word in k}
    sorted_word_tags = sorted(word_tags.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_word_tags)>0:
        return sorted_word_tags[0]
    else:
        return None

from nltk.tokenize import word_tokenize
predictedBaseline = []
# Prediksi menggunakan seluruh kata dalam testSet
for i in range (len(testSet)):
    token = str(testSet.iloc[i,0]).lower()
    most_freq_tag_token = get_most_freq_tag_word(token)
    if most_freq_tag_token:
        # kata terdapat di data latih
        predictedBaseline.append(most_freq_tag_token[0][1])
    else:
        # kata tidak terdapat di data latih
        predictedBaseline.append(get_most_freq_tag())

dfPredictedBaseline = copy.deepcopy(testSet)
dfPredictedBaseline = dfPredictedBaseline.rename(columns={0: "Kata", 1: "Tag"})
dfPredictedBaseline.insert(2, "Predicted Tag", predictedBaseline)
# dfPredictedBaseline.to_excel("Hasil Baseline.xlsx", index =  False)

accuracyBaseline = 0
for i in range (len(dfPredictedBaseline)):
    if dfPredictedBaseline.iloc[i,1] == dfPredictedBaseline.iloc[i,2]:
        accuracyBaseline += 1
accuracyBaseline = accuracyBaseline / len(dfPredictedBaseline)
print("Akurasi: ", accuracyBaseline)