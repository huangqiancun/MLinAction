from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
"""
函数说明：创建数据集
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性
Modify:
    2018-07-29
"""
def createDateSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels

"""
函数说明：计算给定数据集的经验熵(香农熵)
Parameters:
    dataSet-数据集
Returns:
    shannonEnt-数据集
Modify:
    2018-07-29
"""
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntires
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

"""
函数说明：按照给定特征划分数据集
Parameters:
    dataSet-待划分的数据集
    axis-划分数据集的特征
    value-需要返回的特征的值
Returns:
    retDataSet-划分后的数据集
Modify:
    2018-07-29
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = [] #创建返回的数据集
    for featVec in dataSet: #遍历数据集
        if featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]  #去掉axis特征
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet

"""
函数说明：选择最优特征
Parameters:
    dataSet-数据集
Returns:
    bestFeature-信息增益最大的特征的索引值
Modify:
    2018-07-29
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #特征数量
    baseEntropy = calcShannonEnt(dataSet) #计算数据集的香农熵
    bestInfoGain = 0.0  #信息增益
    bestFeature = -1 #最优特征的索引值
    for i in range(numFeatures): #遍历所有特征
        #获取dataSet的第i列的所有行
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) #创建set集合{}，元素不可重复
        newEntropy = 0.0 #经验条件熵
        for value in uniqueVals: #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/ float(len(dataSet))
            newEntropy += prob*calcShannonEnt((subDataSet))
        infoGain = baseEntropy - newEntropy
        #print("第%d个特征的增益为%.3f"%(i, infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
"""
函数说明：统计classList中出现此处最多的元素（类标签）
Parameters:
    classList-类标签列表
Returns:
    sortedClassCount[0][0] -出现此处最多的元素（类标签）
Modify:
    2018-07-30
"""
def majorityCnt(classList):
    classCount = {} #统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.keys(),key = operator.itemgetter,reverse=True) #根据字典的值降序排序
    return sortedClassCount[0][0]

"""
函数说明：创建决策树
Parameters:
    dataSet-训练数据集
    labels-分类属性标签
    featLabels-存储选择的最优特征标签
Returns:
    myTree-决策树
Modify:
    2018-07-30
"""
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet] #取分类标签
    if classList.count(classList[0]) == len(classList): #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1: #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  #选择最优特征
    bestFeatLabel = labels[bestFeat] #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}   #根据最优特征的标签生成树
    del(labels[bestFeat]) #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]   #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  #去掉重复的属性值
    for value in uniqueVals: #遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
    return myTree

"""
函数说明：获取决策树种叶子结点的数目
Parameters:
    myTree-决策树
Returns:
    numLeafs-决策树的叶子结点的数目
Modify:
    2018-07-31
"""
def getNumLeafs(myTree):
    numLeafs = 0 #初始化叶子
    firstStr = next(iter(myTree)) #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr] #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]) is dict: #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
                numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

"""
函数说明：获取决策树的层数
Parameters:
    myTree-决策树
Returns:
    maxDepth-决策树的层数
Modify:
    2018-07-31
"""
def getTreeDepth(myTree):
    maxDepth = 0 #初始化决策树深度
    firstStr = next(iter(myTree)) #
    secondDict = myTree[firstStr] #获取下一个字典
    thisDepth = 0
    for key in secondDict.keys():
        if type(secondDict[key]) is dict: #测试该节点是否字典
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth += 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth #更新层数
    return maxDepth
"""
函数说明：绘制节点
Parameters:
    nodeTxt-节点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 节点格式
Returns:
    无
Modify:
    2018-07-31
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle = "<-") #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size = 14)#设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

"""
函数说明：标注有向边属性值
Parameters:
    cntrPt，parentPt-用于计算标注位置
    txtString-标注的内容
Returns:
    无
Modify:
    2018-07-31
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString,va="center",ha="center",rotation=30)

"""
函数说明：绘制决策树
Parameters:
    myTree-决策树（字典）
    parentPt-标注的内容
    nodeTxt-节点名
Returns:
    无
Modify:
    2018-07-31
"""
def plotTree(myTree,parentPt,nodeTxt):
    decisionNode = dict(boxstyle="sawtooth",fc="0.8")
    leafNode = dict(boxstyle="round4",fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]) is dict:
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff+1.0/plotTree.totalD


"""
函数说明：创建绘图面板
Parameters:
    inTree-决策树（字典）
Returns:
    无
Modify:
    2018-07-31
"""
def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


"""
函数说明：使用决策树分类
Parameters:
    inputTree-已经生成的决策树
    featLabels-存储选择的最优特征标签
    testVec-测试数据列表，顺序对应最优特征标签
Returns:
    classLabel-分类结果
Modify:
    2018-08-01
"""
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) is dict:
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel =secondDict[key]
    return classLabel

"""
函数说明：存储决策树
Parameters:
    inputTree-已经生成的决策树
    filename-决策树存储文件名
Returns:
    无
Modify:
    2018-08-05
"""
import pickle
def storeTree(inputTree, filename):
    with open(filename,'wb') as fw:
        pickle.dump(inputTree, fw)

"""
函数说明：读取决策树
Parameters:
    filename-决策树存储文件名
Returns:
    pickle.load(fr)-决策树字典
Modify:
    2018-08-05
"""
def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)

"""
函数说明：main函数
"""
if __name__ == '__main__':
    # dataSet, labels = createDateSet()
    # shannonEnt = calcShannonEnt(dataSet)
    # featLabels = []
    # myTree = createTree(dataSet,labels, featLabels)
    myTree = {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
    # print(myTree)
    # testVec = [0,1]
    # result = classify(myTree,featLabels,testVec)
    # if result == 'yes':
    #     print('放贷')
    # if result == 'no':
    #     print('不放贷')
    storeTree(myTree,'classifierStorage.txt')
    MyTree = grabTree('classifierStorage.txt')
    print(myTree)
