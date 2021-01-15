import os
import nltk
import math
import copy
import numpy as np 
import pandas as pd
import math
import re
import heapq
from openpyxl import Workbook
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory



def loadDataset():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    factoryStop = StopWordRemoverFactory()              #Stopword
    stopword = factoryStop.create_stop_word_remover()   #Stopword
    path = 'Dataset'
    # Nama Folder
    files = os.listdir(path)
    allArticles = []
    for file in files:
    # Load all file
        f = open(path + "/" + file,  'r', encoding='utf-8')
        # print(file)
        line = f.readlines()
        allArticles.extend(line)   
    newAllAtricle = []
    for allArticle in allArticles:
        newString = allArticle.lower()
        newString = stemmer.stem(allArticle)    #Stemming
        newString = stopword.remove(newString)  #Stopword
        newString = re.sub(r'\W',' ',newString)
        newString = re.sub(r'\s+',' ',newString)
        newAllAtricle.append(newString)
    return newAllAtricle

def getTokens(newAllAtricle):
    wordfreq = {}
    for sentence in newAllAtricle:
        sTokens = nltk.word_tokenize(sentence)
        for token in sTokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    tokens = wordfreq
    #Pencarian kata unik
    # tokens = heapq.nlargest(1000, wordfreq, key=wordfreq.get) # setting wordfreq bisa diubah sesuai dengan karakteristik data yang digunakan. Di sini akan digunakan frekuensi maksimum 200.
    return tokens

def createTFIDF(tokens, article):
    wordDict = []
    for i in range(len(article)):
        wordDict.append(dict.fromkeys(tokens,0))
    for i in range(len(wordDict)):
        checkThis = nltk.word_tokenize(article[i])
        for tokenWord in checkThis:
            if tokenWord in tokens:                     
                wordDict[i][tokenWord] += 1
    #Perhitungan TF
    tfDict = copy.deepcopy(wordDict)
    for i in range (len(article)):
        tokenfromSentence = nltk.word_tokenize(article[i])
        for word in tokenfromSentence:
            if word in tokens:                          
                if tfDict[i][word] != 0:
                    tfDict[i][word] = tfDict[i][word] / len(tokenfromSentence)
                    #rumus
    #Perhitungan IDF
    idfDict = dict.fromkeys(tokens,0)
    for sentence in wordDict:
        for word, val in sentence.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(len(article)/float(val))
    #Perhitungan TFIDF
    TFIDFMatrix = copy.deepcopy(tfDict)
    counter = 0
    for item in TFIDFMatrix:
        for word, val in item.items():
            TFIDFMatrix[counter][word] = tfDict[counter][word]*idfDict[word]
        counter += 1
    TFIDFMatrix = pd.DataFrame(TFIDFMatrix)
    # TFIDFMatrix.to_excel("TFIDFMatrix.xlsx")
    return TFIDFMatrix

def createCoOccurence(tokens, article):
    # Referensi pengerjaan https://web.stanford.edu/~jurafsky/slp3/6.pdf Bagian 6.3.2
    newTokens = []
    for item in article:
        newTokens.extend(nltk.word_tokenize(item.lower()))
    termMatrix = pd.DataFrame(0, columns = tokens, index = tokens)
    for item in termMatrix:
        idxToken = ([i for i,val in enumerate(newTokens) if val==item])
        for idx in idxToken:
            counter = 0
            idxNow = idx
            #4 Kata Kiri yang Berdekatan
            while counter != 4 and idxNow > 0:
                counter += 1
                if item in tokens and newTokens[idx-counter] in tokens: 
                    termMatrix[item][newTokens[idx-counter]] += 1
                    idxNow -= 1
            idxNow = idx
            counter = 0
            #4 Kata Kanan yang Berdekatan
            while counter != 4 and idxNow < len(newTokens)-1:
                counter += 1
                if item in tokens and newTokens[idx+counter]in tokens:
                    termMatrix[item][newTokens[idx+counter]] += 1
                    idxNow += 1
    # termMatrix.to_excel("termMatrix.xlsx")
    return termMatrix

def createPPMI(termMatrix):
    PPMIMatrix = copy.deepcopy(termMatrix)
    PPMIMatrix = PPMIMatrix.astype(float)
    total = 0
    for item in list(PPMIMatrix):
        total += (int(pd.Series(PPMIMatrix[item].sum(), index = [item])))
    for col in PPMIMatrix.columns:
        for idx in PPMIMatrix.index:
            if total != 0:
                PPMIMatrix[col][idx] = PPMIMatrix[col][idx]/total
            else:
                PPMIMatrix[col][idx] = 0
    newPPMIMatrix = copy.deepcopy(PPMIMatrix)
    listPW = [] 
    for col in PPMIMatrix.columns:
        listPW.append(PPMIMatrix[col].sum())
        #Perhitungan P(w)
    newPPMIMatrix.insert((len(newPPMIMatrix)), 'p(w)',listPW )
    #insert P(W)
    listPW.append(sum(listPW))
    newPPMIMatrix.loc['c(w)'] = listPW
    #Insert c(w)
    #P(w) dan C(W) akan sama ketika matriks yang dibangun menggunakan elemen yang sama
    # newPPMIMatrix.to_excel("newPPMIMatrix.xlsx")
    return newPPMIMatrix

def calculatenotZeros(matrix):
    counter = 0
    zeros = 0
    for col in matrix.columns:
        for idx in matrix.index:
            if matrix[col][idx] > 0:
                zeros += 1
            counter += 1
    return str( round((zeros/counter) * 100, 2))+ "% Tidak bernilai Nol"

def calculateCosineDocument(typeMatrix, word1, word2):
    return np.dot(typeMatrix.loc[word1],typeMatrix.loc[word2])/(np.linalg.norm(typeMatrix.loc[word1])*np.linalg.norm(typeMatrix.loc[word2]))

def calculateCosineWord(typeMatrix, word1, word2):
    return np.dot(typeMatrix[word1],typeMatrix[word2])/(np.linalg.norm(typeMatrix[word1])*np.linalg.norm(typeMatrix[word2]))

def calculatePPMI(matrix, word1, word2):
    pwc = matrix[word1][word2]
    pw  = matrix[word1]['c(w)']
    cw  = matrix['p(w)'][word2]
    pwcw = pw*cw
    if pwc > 0.0:
        calculate = math.log(pwc/pwcw,2)
        #Rumus pada jurnal jurafsky
    else:
        calculate = 0
        #Dikarenakan tidak ada implementasi laplace smoothing untuk pwc yang bernilai 0 maka hasilnya akan diassing 0
    return calculate
        

def main():
    newAllAtricle = loadDataset()
    #load seluruh artikel
    tokens = getTokens(newAllAtricle)
    print(len(tokens))
    TFIDFMatrix = createTFIDF(tokens, newAllAtricle)
    termMatrix  = createCoOccurence(tokens, newAllAtricle)
    PPMIMatrix  = createPPMI(termMatrix)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    print("Start Nomor 1")
    print(calculatenotZeros(TFIDFMatrix))
    print(calculatenotZeros(PPMIMatrix))
    print("End Nomor 1\n")
    
    print("Start Nomor 2")
    print("Persamaan antara dokumen 5 dan 5 (Dokumen Sama): ", calculateCosineDocument(TFIDFMatrix, 4, 4))
    print("Persamaan antara dokumen 12 dan 20 (Dokumen dengan Topik Sama): ", calculateCosineDocument(TFIDFMatrix, 11, 19))
    print("Persamaan antara dokumen 3 dan 13  (Dokumen dengan Topik Berbeda): ", calculateCosineDocument(TFIDFMatrix, 2, 12))
    print("End Nomor 2\n")

    print("Perbandingan Kata (Staycation dan Liburan)")
    print("Implementasi Matriks TF-IDF: ", calculateCosineWord(TFIDFMatrix, stemmer.stem('staycation'), stemmer.stem('liburan')))
    print("Implementasi Matriks Term Context: ", calculateCosineWord(termMatrix, stemmer.stem('staycation'), stemmer.stem('liburan')))
    print("Implementasi Matriks PPMI: ", calculatePPMI(PPMIMatrix,stemmer.stem('staycation'), stemmer.stem('liburan')))
    
    print("Perbandingan Kata (Staycation dan Kebakaran)")
    print("Implementasi Matriks TF-IDF: ", calculateCosineWord(TFIDFMatrix, stemmer.stem('staycation'), stemmer.stem('Kebakaran')))
    print("Implementasi Matriks Term Context: ", calculateCosineWord(termMatrix, stemmer.stem('staycation'), stemmer.stem('Kebakaran')))
    print("Implementasi Matriks PPMI: ", calculatePPMI(PPMIMatrix,stemmer.stem('staycation'), stemmer.stem('hutan')))



main()