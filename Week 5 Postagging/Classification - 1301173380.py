import pandas as pd
import nltk
from nltk import word_tokenize
import os
nltk.download('punkt')

path = 'Dataset'
trainSet = pd.read_csv(path + "/Train Set.tsv",header = None, delimiter="\t")
testSet = pd.read_csv(path + "/Test Set.tsv",header = None, delimiter="\t")

kalimatTrain = []
insertKalimat = ''
for i in range(len(trainSet)):
    insertKalimat += str(trainSet.loc[i,0])
    insertKalimat += '/'
    insertKalimat += str(trainSet.loc[i,1])
    insertKalimat += '@'
    if trainSet.loc[i,0] == "." and trainSet.loc[i,1] == "Z":
        kalimatTrain.append(insertKalimat)
        insertKalimat = ''
kalimatTest = []
insertKalimat = ''
for i in range(len(testSet)):
    insertKalimat += str(testSet.loc[i,0])
    insertKalimat += '@'
#     insertKalimat += str(testSet.loc[i,1])
    if testSet.loc[i,0] == "." and testSet.loc[i,1] == "Z":
        kalimatTest.append(insertKalimat)
        insertKalimat = ''

sentences = [] # list untuk menampung kalimat dan kata-kata di dalamnya
tags = []    # list untuk menampung tag

for i in range(0,len(kalimatTrain)):
    tokens = kalimatTrain[i].split('@')
    tokens.pop()
    sent = []
    tag = []
    for j in range(0,len(tokens)):
        pair = tokens[j].split('/')
        sent.append(pair[0])
        tag.append(pair[1])
    sentences.append(sent)
    tags.append(tag)

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    #print("sentence index = ")
    #print(sentence[index])
    prefix_1 = ''
    prefix_2 = ''
    suffix_1 = ''
    suffix_2 = ''
    if (len(sentence[index])>2):
      prefix_1 = sentence[index][0]
      prefix_2 = sentence[index][:2]
      suffix_1 = sentence[index][-1]
      suffix_2 = sentence[index][-2:]
    return {
        'word': sentence[index],
        'prefix-1': prefix_1,
        'prefix-2': prefix_2,        
        'suffix-1': suffix_1,
        'suffix-2': suffix_2,        
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    }

def transform_to_dataset(sentences, tags):
    X, y = [], []
 
    for sentence_idx in range(len(sentences)):
        for index in range(len(sentences[sentence_idx])):
            X.append(features(sentences[sentence_idx], index))
            y.append(tags[sentence_idx][index])
 
    return X, y

cutoff = int(.75 * len(sentences))
training_sentences = sentences[:cutoff]
test_sentences = sentences[cutoff:]
training_tags = tags[:cutoff]
test_tags = tags[cutoff:]

X, y = transform_to_dataset(training_sentences, training_tags)


from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
 
clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', tree.DecisionTreeClassifier(criterion='entropy'))
])
clf.fit(X, y)   
 
print('Training completed')

def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return tags 

predictedClassification = []
for i in range (len(kalimatTest)):
    predicting = []
    predicting.extend(pos_tag(kalimatTest[i].split('@')))
    predicting.pop()
    predictedClassification.extend(predicting)
    
import copy
dfPredictedClassification = copy.deepcopy(testSet)
dfPredictedClassification = dfPredictedClassification.rename(columns={0: "Kata", 1: "Tag"})
dfPredictedClassification.insert(2, "Predicted Tag", predictedClassification)
# dfPredictedClassification.to_excel("Hasil Classification.xlsx", index =  False)

accuracyClassification = 0
for i in range (len(dfPredictedClassification)):
    if dfPredictedClassification.iloc[i,1] == dfPredictedClassification.iloc[i,2]:
        accuracyClassification += 1
accuracyClassification = accuracyClassification / len(dfPredictedClassification)
print("Akurasi: ", accuracyClassification)