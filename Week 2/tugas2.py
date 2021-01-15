import os
import nltk

def loadArticle():
    path = 'Staycation'
    # Nama Folder
    files = os.listdir(path)
    print("Mulai Load Data")
    allArticle = []
    for file in files:
    # Load all file
        article = []
        f = open(path + "/" + file,  'r', encoding='utf-8')
        line = f.readline()
        allArticle.append(line)
    # Memasukan seluruh artikel ke list
    newAllAtricle = []
    for allArticles in allArticle:
        newString = allArticles.replace("\n", "")
        newString = newString.replace(".", "")
        newString = newString.replace(",", "")
        newString = newString.replace("–", "")
        newString = newString.replace("@", "")
        newString = newString.lower()
    #Penghilangan tanda baca
        newAllAtricle.append(newString)
    print("Load Data Selesai")
    return newAllAtricle

def preprocess(newAllAtricle):
    # Preprocess
    sentencedTokens = []
    for articles in newAllAtricle:
        sent_text = nltk.sent_tokenize(articles) # tokenisasi menjadi beberapa kalimat
        # loop per kalimat, tambahkan tag <s> di awal kalimat dan </s> di akhir kalimat
        mod_sentences = [] # array/list untuk menampung semua kalimat teks awal
        for sentence in sent_text:
            mod_sentence = []
            sent_tokens = nltk.word_tokenize(sentence)
            mod_sentence.append('<s>')
            for token in sent_tokens:
                mod_sentence.append(token)
            mod_sentence.append('</s>')
            mod_sentences.append(mod_sentence)
        sentencedTokens.extend(mod_sentences)
    return sentencedTokens

# Unigram
def createUniFreqTab(tokens):
    freq_tab = {}
    for sentence in sentencedTokens:
        for token in sentence:
            if token in freq_tab:
                freq_tab[token] += 1 # kata sudah ada di dictionary, update frekuensinya
            else:
                freq_tab[token] = 1  # kata belum ada di dictionary 
    return freq_tab

# Bigram
def createBiFreqTab(tokens):
    freq_bigram_tab = {}
    for sentence in sentencedTokens:
          for i in range (1, len(sentence)):
            curr_bigram = (sentence[i-1], sentence[i])
            if curr_bigram in freq_bigram_tab:
                  freq_bigram_tab[curr_bigram] += 1 # bigram sudah ada di dictionary, update frekuensinya
            else:
                  freq_bigram_tab[curr_bigram] = 1 # bigram belum ada di dictionary
    return freq_bigram_tab

def createBiProbTab(freqTabBi, freqTabUni):
    bigram_prob_tab = {}
    for sentence in sentencedTokens:
        for i in range (1, len(sentence)):
            curr_bigram = (sentence[i-1], sentence[i])
            if curr_bigram not in bigram_prob_tab:  
                bigram_prob_tab[curr_bigram] = freqTabBi[curr_bigram]/freqTabUni[sentence[i-1]] # bigram belum ada di dictionary, hitung probability  
    return bigram_prob_tab

def biLaplaceSmoothing(newWords, freqTabBi, probTabBi):
    newFreqTab = freqTabBi.copy()
    newProbTab = probTabBi.copy()
#   List untuk menampung Frekuensi dan Probability
    for item in newFreqTab:
        newFreqTab[item] += 1 #Menambahkan seluruh item dengan +1
    for word in newWords:
        newFreqTab[word] = 1  #Menambahkan bigram baru kedalam dict
    for item in newFreqTab:
        newProbTab[item] = newFreqTab[item]/(sum(newFreqTab.values()))
#         Membuat Probability Bigram
    return newProbTab

def perplexity(freqBi, sentence):
    x = 1.0
    for i in range(len(sentence)):
        x = x*(1/freqBi[sentence[i]])                        
    l = x**(1/len(sentence))
    return(l)


def testSentenceBigram(probBi, sentence):
    lc_test_sentence = sentence.lower()
    test_tokens = ['<s>']
    test_tokens.extend(nltk.word_tokenize(lc_test_sentence))
    test_tokens.append('</s>')
    total_prob = 1.0
    print(test_tokens)
    checkThis = []
    for i in range(1, len(test_tokens)):
        checkThis.append((test_tokens[i-1],test_tokens[i]))
#   Membuat list bigram untuk kalimat uji
    checkLaplace = False

    newWords = []
    for test_token in checkThis:
        if test_token not in probBi:
            newWords.append(test_token)
            checkLaplace = True
#   Mencari bigram yang tidak ada dalam data latih

    if not checkLaplace:
        probTab = probBi.copy()
#   Ketika kalimat uji tidak memerlukan laplace smoothing
    else:
        probTab = biLaplaceSmoothing(newWords, freqBi, probBi)
#   Ketika kalimat uji memerlukan laplace smoothing

    for test_token in checkThis:
        total_prob = total_prob * probTab[test_token]
        
    print("Probability: ", total_prob)
    print("Perplexity: " ,  round(perplexity(probTab, checkThis)))
#     print("Test Perplex: ", round(testPerplexity(total_prob, checkThis),4))

#Trigram
def createFreqTrigram(tokens):
    freq_trigram_tab = {}
    for sentence in tokens:
          for i in range (2, len(sentence)):
            curr_trigram = (sentence[i-2], sentence[i-1], sentence[i])
            if curr_trigram in freq_trigram_tab:
                  freq_trigram_tab[curr_trigram] += 1 # trigram sudah ada di dictionary, update frekuensinya
            else:
                  freq_trigram_tab[curr_trigram] = 1 # trigram belum ada di dictionary
    return freq_trigram_tab

newAllAtricle = loadArticle()
sentencedTokens = preprocess(newAllAtricle)
freqUni = createUniFreqTab(sentencedTokens)
freqBi = createBiFreqTab(sentencedTokens)
probBi = createBiProbTab(freqBi, freqUni)
print(testSentenceBigram(probBi, "Staycation adalah cara liburan yang cenderung hemat"))
print(testSentenceBigram(probBi, "Staycation adalah kegiatan berlibur di dekat rumah"))
print(testSentenceBigram(probBi, "Staycation alternatif liburan yang paling diminati warga jakarta"))
print(testSentenceBigram(probBi, "Facebook merupakan sosial media dengan pengguna tertinggi di dunia"))
print(testSentenceBigram(probBi, "Pembelian sepeda meningkat pesat pada saat pandemi COVID"))
print(testSentenceBigram(probBi, "PSBB merupakan salah satu jalan yang ditempuh pemerintah Indonesia"))


print("\n\nBonus    ")
choice = input("Apakah ingin melihat freq dari Trigram? [Y/N] : ")
choice = choice.lower()
if choice == "y":
    print(createFreqTrigram(sentencedTokens))
else:
    print("Terima Kasih")

