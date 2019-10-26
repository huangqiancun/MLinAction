from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    """
    加载数据
    :param
        fileName:文件名
    :return:
        xArr:x数据集
        yArr:y数据集
    Modify：
        2018-8-18
    """
    numFeat = len(open(fileName).readline().split('\t'))-1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr,yArr

def regularize(xMat,yMat):
    """
    数据标准化
    :param
        xMat:x数据集
        yMat:y数据集
    :return:
        inxMat:标准化后的x数据集
        inyMat:标准化后的y数据集
    """
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    yMean = np.mean(inxMat,0)
    inyMat = yMat-yMean
    inMeans = np.mean(inxMat,0)
    inVar = np.var(inxMat,0)
    inxMat = (inxMat-inMeans)/inVar
    return inxMat,inyMat

def rssError(yArr,yHatArr):
    """
    计算平均误差
    :param
        yArr:真实值
        yHatArr: 预测值
    :return: 误差
    """
    return ((yArr-yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    """
    前向逐步线性回归
    :param
        xArr:x输入数据
        yArr:y预测数据
        eps:每次迭代需要调整的步长
        umIt:迭代次数
    :return:
        returnMat：numIt次迭代的回归系数矩阵
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat,yMat = regularize(xMat,yMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMat = ws.copy()
    for i in range(numIt):
        lowestError = float('inf')
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError = rssE
                    wsMat = wsTest
        ws = wsMat.copy()
        returnMat[i,:] = ws.T
    return returnMat

def plotstageWiseMat():
    """
    绘前向逐步回归系数矩阵
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet('abalone.txt')
    returnMat = stageWise(abX, abY,0.005,1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归：迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='red')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='red')
    plt.show()

if __name__=='__main__':
    plotstageWiseMat()