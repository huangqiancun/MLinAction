# -*_ coding: UTF-8 -*-
import re
import numpy as np
import random
"""
函数说明：接受一个大字符串并将其解析为字符串列表
parameters：
    无
returns：
    无
Modify：
    2018-8-8
"""
def textParse(bigString):# 将字符串转换为字符列表
    listOfTokens = re.split(r'\W+',bigString) #将特殊符号作为切分标志进行字符串切分，即非字母，非数字
    return [tok.lower() for tok in listOfTokens if len(tok)>2] #除了单个字母，例如大写的I，其他单词变成小写
"""
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
parameters：
    dataSet-整理的样本数据集
returns：
    vocabSet-返回不重复的词条列表，也就是词汇表
Modify：
    2018-8-8
"""
def createVocabList(dataSet):
    vocabSet = set([])#创建一个空的不重复的列表
    for document in dataSet:
        vocabSet = vocabSet|set(document)#取并集
    return list(vocabSet)
"""
函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
parameters：
    vocabList-createVocabList返回的列表
    inputSet-切分的词条列表
returns：
    returnVec-文档向量，词集模型
Modify：
    2018-8-8
"""
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:#遍历每个词条
        if word in vocabList:#如果词条存在词汇表中，置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the world: %s is not in my vocabulary!" %word)
    return returnVec

"""
函数说明：根据vocabList词汇表，构建词袋模型
parameters：
    vocabList-createVocabList返回的列表
    inputSet-切分的词条列表
returns：
    returnVec-文档向量，词袋模型
Modify：
    2018-8-9
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

"""
函数说明：朴素贝叶斯分类器训练函数
Parameters：
    trainMatrix-训练文本矩阵，即setOfWords2Vec返回的returnVec矩阵
    trainCategory-训练类别标签向量，即loadDataSet返回的classVec
Returns：
    p0Vect-非侮辱类的条件概率数组
    p1Vect-侮辱类的条件概率数组
    pAbusive-文档属于侮辱类的概率
Modify：
    2018-8-5
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#计算训练的文档数目
    numWords = len(trainMatrix[0])#计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)#文档属于侮辱类的概率
    p0Num = np.ones(numWords)#词条出现数初始化为1
    p1Num = np.ones(numWords)
    p0Denom = 2.0#分母初始化为2,拉普拉斯平滑
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:#统计属于侮辱类的条件概率所需的数据，即p(w0|1),p(w1|1)
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:#统计属于非侮辱类的条件概率所需的数据，即p(w0|0),p(w1|0)
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive  #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

"""
函数说明：朴素贝叶斯分类器分类函数
Parameters：
    vec2Classify-待分类的词条数组
    p0Vec-侮辱类的条件概率数组
    p1Vec-非侮辱类的条件概率数组
    p1Class1-文档属于侮辱类的概率
Returns：
    0-属于非侮辱类
    1-属于侮辱类
Modify：
    2018-8-5
"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

"""
函数说明：测试朴素贝叶斯分类器
Parameters：
    无
Returns：
    无
Modify：
    2018-8-9
"""
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26): #分别读取25个txt文件
        wordList = textParse(open('./email/spam/%d.txt'% i, 'r').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('./email/ham/%d.txt'% i, 'r').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#创建词汇表，不重复
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集:",docList[docIndex])
    print("错误率：%.2f%%"%(float(errorCount)/len(testSet)*100))


"""
函数说明：main函数
"""
if __name__=='__main__':
    spamTest()