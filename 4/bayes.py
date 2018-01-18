# _*_ coding:utf-8 _*_  //为了告诉python解释器，按照utf-8编码读取源代码，否则，你在源代码中写的中文输出可能会由乱码  

from numpy import *

def loadDataSet():#样本向量
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 为侮辱性, 0 not
    return postingList,classVec

def createVocabList(dataSet):#postingList为参数，返回 词集合
    vocabSet = set([])  #create a empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets，集合求并
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):#输出为 词向量，表示改词是否出现，词集模型,输入：vocabSet，lpostingList
    returnVec = [0]*len(vocabList)#等长的0
    for word in inputSet:
        if word in vocabList:#出现了词汇集合中的单词
            returnVec[vocabList.index(word)] = 1#标志置1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):#词袋模型，词向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1#标志+1
    return returnVec

def trainNB0(trainMatrix,trainCategory):#输出三个概率，2个集合，一个数值，输入：词向量，classVec
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):#计算概率，判断类型，输入：词向量
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
#------------------------------分类器构建完毕--------------------------------------------#


#------------------------------过滤恶意留言--------------------------------------------#
def testingNB():#main：输入新的词句，判断类型
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
#------------------------------完毕--------------------------------------------#


#------------------------------邮件过滤--------------------------------------------#
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)#qie fen
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

def spamTest():#邮件过滤
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())#读入文本&&分割(spam)
        docList.append(wordList)#doc [[],[]...]全部词
        fullText.extend(wordList)#full [...]
        classList.append(1)#class[1,1,1,1,1,1,1,...]，label
        wordList = textParse(open('email/ham/%d.txt' % i).read())#读入文本&&分割(ham)
        docList.append(wordList)#doc
        fullText.extend(wordList)#full
        classList.append(0)#class[0,0,0，0，0,0,0...]，
    vocabList = createVocabList(docList)#create vocabulary 词集合列表

    trainingSet = range(50); testSet=[] #create test set
    for i in range(10):#随机选出测试集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) 
         
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0训练
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))#词向量（对应第docIndex个label）
        trainClasses.append(classList[docIndex])#label
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))#3个概率数组
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items测试
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])#词向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#贝叶斯计算概率
            errorCount += 1 #错误个数
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText
#------------------------------完毕--------------------------------------------#


#------------------------------广告中获取区域倾向--------------------------------------------#
def calcMostFreq(vocabList,fullText):#对词列表中的的出现次数排序，取前30个
    import operator
    freqDict = {}
    for token in vocabList:#对每个词句
        freqDict[token]=fullText.count(token)#在fullText中的出现次数
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) #从大到小
    return sortedFreq[:30] #返回出现次数最高的30个词      

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #0
    vocabList = createVocabList(docList)#create vocabulary，词（切割后的句子）的集合列表

    top30Words = calcMostFreq(vocabList,fullText)   #在vocabList remove top 30 words 30个词句子
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])

    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  

    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))#词向量
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items，判断这是那个地域的？？？
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])#词向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):#显示地域相关用词
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):#对于词向量中的每一个次数
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)#按照概率排序
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]












