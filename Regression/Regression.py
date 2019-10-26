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

def standRegression(xArr,yArr):
    """
    计算回归系数w
    :param
        xArr:x数据集
        yArr:y数据集
    :return:
        ws：回归系数
    ：Modify：
        2018-8-18
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat #根据文中推导公式计算回归系数
    if np.linalg.det(xTx) == 0:
        print('矩阵为奇异矩阵，不能求逆')
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

def plotRegression():
    """
    绘制回归曲线和数据点
    :param:
        无
    :return:
        无
    ：Modify：
        2018-8-18
    """
    xArr,yArr = loadDataSet('ex0.txt')#加载数据集
    ws = standRegression(xArr,yArr)#计算回归系数
    xMat = np.mat(xArr)#创建矩阵
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha = .5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


def plotDataSet():
    """
    绘制数据集
    :param：
        无
    :return:
        无
     Modify：
        2018-8-18
    """
    xArr,yArr = loadDataSet('ex0.txt')
    n = len(xArr)
    xcord = []
    ycord = []
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='blue',alpha = 0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__=='__main__':
    xArr,yArr = loadDataSet('ex0.txt')#加载数据集
    ws=standRegression(xArr,yArr)#计算回归系数
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws
    print(np.corrcoef(yHat.T,yMat))