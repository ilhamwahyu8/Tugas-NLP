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
    insertKalimat += str(trainSet.loc[i,0]).lower()
    insertKalimat += '/'
    insertKalimat += str(trainSet.loc[i,1])
    insertKalimat += '@'
    if trainSet.loc[i,0] == "." and trainSet.loc[i,1] == "Z":
        kalimatTrain.append(insertKalimat)
        insertKalimat = ''
kalimatTest = []
insertKalimat = ''
for i in range(len(testSet)):
    insertKalimat += str(testSet.loc[i,0]).lower()
    insertKalimat += '@'
#     insertKalimat += str(testSet.loc[i,1])
    if testSet.loc[i,0] == "." and testSet.loc[i,1] == "Z":
        kalimatTest.append(insertKalimat)
        insertKalimat = ''

def viterbi(transition_prob, emission_prob, tokens):
    # cek apakah semua token ada di vocab/emission probability
    oov_status = 0
    counter_check = 0
    stop = False
    while (counter_check < len(tokens)) and (not stop):
        if tokens[counter_check] not in vocabs.values():
            stop = True
            oov_status = 1
        else:
            counter_check += 1
    
    if oov_status == 1:
        # kalimat uji mengandung unknown word/OOV, tanpa smoothing tidak bisa diproses
        print('kalimat uji mengandung unknown word')
        return None, None
    else:
        # create a path probability matrix viterbi[N,T]
        # N: banyaknya state
        # T: jumlah token

        T, N = len(tokens)+1, len(tags) # token ditambah satu untuk start
        new_tokens = ['<s>'] + tokens
        print('T=',T,',N=',N)
        print('token:',new_tokens)
        viterbi_mat = [[0 for x in range(T)] for y in range(N)] 
        # create backpointers matrix
        backpointers = [[0 for x in range(T)] for y in range(N)] 

        # initial probability distribution over states (phi)
        # transition probability dengan previous state adalah <start>
        phi = {}
        for i in range (1,len(tags)):
            phi[tags[i]] = transition_prob[('<start>',tags[i])]
                    
        # initialization
        # urutan index state sesuai dengan index di dictionary tags{}
        # inisialisasi dimulai dari state ke-1, state ke-0 sudah pasti <start>
        viterbi_mat[0][0] = 1.0 # untuk token <s>, tag = <start>
        for s in range(1,N):
            viterbi_mat[s][1] = phi[tags[s]] * emission_prob[(tags[s],new_tokens[1])]
            backpointers[s][1] = 0


        # recursion step
        for t in range(2,T):
            for s in range(1,N):
                # get max viterbi from previous transition
                max_prev_transition = 0.0
                max_state = 0
                for i in range(1,N):                
                    #selain transisi dari tag <start>, range mulai dari indeks 1
                    temp_transition = viterbi_mat[i][t-1] * transition_prob[(tags[i],tags[s])]
                    if temp_transition > max_prev_transition:
                        max_prev_transition = temp_transition
                        max_state = i
                viterbi_mat[s][t] = max_prev_transition * emission_prob[(tags[s],new_tokens[t])]
                backpointers[s][t] = max_state

        # terminasi
        # get max probability in last column
        max_last_prob = 0.0
        best_last_tag = ''
        idx_best_last_tag = 0
        for i in range (1,N):
            if viterbi_mat[i][T-1] > max_last_prob:
                max_last_prob = viterbi_mat[i][T-1]
                best_last_tag = tags[i]
                idx_best_last_tag = i

        best_path = []
        best_path.append(idx_best_last_tag)
        for i in range(T-1,1,-1):
            best_prev_tag = backpointers[idx_best_last_tag][i]
            best_path.append(best_prev_tag)
        # reverse the order
        best_path = best_path[::-1]
    
        return viterbi_mat, best_path

vocabs = {}
for i in range(len(trainSet)):
    vocabs[i] = str(trainSet.iloc[i,0]).lower()

tags = {}
counter = 0
for i in range(len(trainSet)):
    if trainSet.loc[i,0] == "." and trainSet.loc[i,1] == "Z":
        tags[counter] = str(trainSet.iloc[i,1])
        counter += 1
        tags[counter] = '<start>'
    else:
        tags[counter] = str(trainSet.iloc[i,1])
    counter += 1

possibilityWord = []
for i in range (len(trainSet)):
    possibilityWord.append(trainSet.loc[i,0].lower())
possibilityWord = set(possibilityWord) 
possibilityWord = (list(possibilityWord)) 
# len(possibilityWord)

possibilityTag = []
for i in range (len(tags)):
    possibilityTag.append(tags[i])
possibilityTag = set(possibilityTag) 
possibilityTag = (list(possibilityTag))
# len(possibilityTag)

emissionProb = {}
# possibilityTagEP = copy.deepcopy(possibilityTag)
# possibilityTagEP.remove('<start>')
for tag in possibilityTag:
    for word in possibilityWord:
        emissionProb[tag,word] = 0
for i in range (len(trainSet)):
    check = str(trainSet.loc[i,1]), str(trainSet.loc[i,0]).lower()
    if check in emissionProb:
        emissionProb[str(trainSet.loc[i,1]), str(trainSet.loc[i,0]).lower()] += 1  
import copy
emissionProbClc = copy.deepcopy(emissionProb)
for tag, word in emissionProb:
    total = 0
    for wordd in possibilityWord:
        total += emissionProbClc[tag, wordd]
    if total == 0:
        emissionProb[tag, word] = 0
    else:
        emissionProb[tag, word] = emissionProb[tag, word] / total
# len(emissionProb)

transitionProb = {}
for tag1 in possibilityTag:
    for tag2 in possibilityTag:
        transitionProb[tag1,tag2] = 0
# transitionProb
for i in range (1, len(tags)):
    tagXtag = tags[i-1], tags[i]
    if tagXtag in transitionProb:
        transitionProb[tagXtag] += 1
import copy
transitionProbClc = copy.deepcopy(transitionProb)
for tag1, tag2 in transitionProb:
    total = 0
    for tag in possibilityTag:
        total += transitionProbClc[tag1, tag]
    transitionProb[tag1, tag2] = transitionProb[tag1, tag2] / total
# len(transitionProb)

predictedHMM = []
for i in range (len(kalimatTest)):
    checkThis = kalimatTest[i].split('@')
    checkThis.pop()
    viterbi_mat, best_path =  viterbi(transitionProb, emissionProb, checkThis)
    for tag in best_path:
        predictedHMM.append(tags[tag])

import copy
dfPredictedHMM = copy.deepcopy(testSet)
dfPredictedHMM = dfPredictedHMM.rename(columns={0: "Kata", 1: "Tag"})
dfPredictedHMM.insert(2, "Predicted Tag", predictedHMM)
# dfPredictedHMM.to_excel("Hasil HMM.xlsx", index =  False)

accuracyHMM = 0
for i in range (len(dfPredictedHMM)):
    if dfPredictedHMM.iloc[i,1] == dfPredictedHMM.iloc[i,2]:
        accuracyHMM += 1
accuracyHMM = accuracyHMM / len(dfPredictedHMM)
print("Akurasi: ", accuracyHMM)