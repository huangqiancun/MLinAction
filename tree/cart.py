import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    加载数据
    :param fileName:文件名
    :return dataMat:数据矩阵
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    """
    根据特征切分数据集合
    :param dataSet: 数据集合
    :param feature: 带切分的特征
    :param value: 该特征的值
    :return:
        mat0-切分的数据集合0
        mat1-切分的数据集合1
    """
    mat0 = dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def plotDataSet(fileName):
    """
    绘制数据集
    :param fileName:文件名
    :return: 无
    """
    dataMat = loadDataSet(fileName)#加载数据集
    n = len(dataMat)#数据个数
    xcord = []#样本点
    ycord = []
    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)#添加subplot
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def regLeaf(dataSet):
    """
    生成叶节点
    :param dataSet:数据集合
    :return: 目标变量的均值
    """
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    """
    误差估计函数
    :param dataSet:数据集合
    :return: 目标变量的总方差
    """
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    """
    找到数据的最佳二元切分方式
    :param dataSet: 数据集合
    :param leafType: 生成叶节点
    :param errType: 误差估计函数
    :param ops: 用于定义的参数构成的元祖
    :return:
        bestIndex-最佳切分特征
        bestValue-最佳特征值
    """
    import types
    #tolS允许的误差下降值，tolN切分的最少样本数
    tolS=ops[0]
    tolN=ops[1]
    #如果当前所有值相等，则退出（根据set的特性）
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None, leafType(dataSet)
    #统计数据集合的行m和列n
    m,n=np.shape(dataSet)
    #默认最后一个特征为最佳切分特征，计算其误差估计
    S=errType(dataSet)
    #分别为最佳误差，最佳特征切分的索引值，最佳特征值
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    #遍历所有的特征列
    for featIndex in range(n-1):
        #遍历所有特征值
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            #根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            #如果数据少于tolN，则退出
            if (np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
                continue
            #计算误差估计
            newS = errType(mat0)+errType(mat1)
            #如果误差估计更小，则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减小不大则退出
    if (S-bestS)<tolS:
        return None,leafType(dataSet)
    #根据最佳的切分特征和特征值切分数据集合
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    #如果切分出的数据集很小则退出
    if (np.shape(mat0)[0]<tolN) or(np.shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    #返回最佳切分特征和特征值
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    """
    数构建函数
    :param dataSet:数据集合
    :param leafType: 建立叶节点函数
    :param errType: 误差计算函数
    :param ops: 包含数构建的所有其他参数的集合
    :return:
        retTree：构建的回归树
    """
    #选择最佳切分特征和切分值
    feat,val =chooseBestSplit(dataSet,leafType,errType,ops)
    #如果没有特征，则返回特征值
    if feat==None:
        return val
    #回归树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #分成左数据集和右数据集
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    #创建左子树和右子树
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

if __name__=='__main__':
    myDat = loadDataSet('ex00.txt')
    myDat = np.mat(myDat)
    print(createTree(myDat))

