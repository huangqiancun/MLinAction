import numpy as np
"""
函数说明：创建实验样本
Parameters：
    无
Returns：
    postingList-实验样本切分的词条
    classVec-类别标签向量
Modify：
    2018-8-5
"""
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

"""
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters：
    dataSet-整理的样本数据集
Returns：
    vocabSet-返回不重复的词条列表，也就是词汇表
Modify：
    2018-8-5
"""
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

"""
函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters：
    vocabList-createVocabList返回的列表
    inputSet-切分的词条列表
Returns：
    returnVec-文档向量
Modify：
    2018-8-5
"""
def setOfWord2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" %word)
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
    print('p0:',p0)
    print('p1:',p1)
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
    2018-8-5
"""
def testingNB():
    postingList,classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(classVec))


    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
    testEntry = ['stupid']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')
"""
函数说明：main函数
parameters：
    无
return：
    无
Modify：
    2018-8-5
"""
if __name__ == '__main__':
    testingNB()