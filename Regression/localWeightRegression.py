import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
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

def lwlr(testPoint,xArr,yArr,k=1.):
    """
    使用局部加权线性回归计算回归系数w
    :param
        testPoint:测试样本点
        xArr:x数据集
        yArr:y数据集
        k:高斯核的k，自定义参数
    :return:
        ws-回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))#创建权重对角矩阵
    for j in range(m):#遍历数据集计算每个样本的权重
        diffMat = testPoint-xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if np.linalg.det(xTx) ==0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I*(xMat.T*(weights*yMat))#计算回归系数
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.):
    """
    局部加权线性回归测试
    :param
        testArr: 测试数据集
        xArr: x数据集
        yArr: y数据集
        k: 高斯核的k，自定义参数
    :return:
        ws-回归系数
    """
    m = np.shape(testArr)[0]#计算测试集的大小
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)#对每个样本点进行预测
    return yHat

def plotlwlrRegression():
    """
    绘制多条局部加权回归曲线
    :return: 无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    xArr,yArr = loadDataSet('ex0.txt')#加载数据集
    yHat_1 = lwlrTest(xArr,xArr,yArr,1.0)#根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)#创建矩阵
    yMat = np.mat(yArr)
    srtInd = xMat[:,1].argsort(0)#排序，返回索引值
    xSort = xMat[srtInd][:,0,:]
    fig,axs = plt.subplots(nrows=3,ncols=1,sharex =False,sharey=False,figsize=(10,8))
    axs[0].plot(xSort[:,1],yHat_1[srtInd],c = 'red')
    axs[1].plot(xSort[:,1],yHat_2[srtInd],c = 'red')
    axs[2].plot(xSort[:,1],yHat_3[srtInd],c = 'red')
    axs[0].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=.5)#绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    #设置标题，xlabel，ylabel
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0',FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.002', FontProperties=font)
    plt.setp(axs0_title_text,size=8,weight='bold',color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()

if __name__=='__main__':
    plotlwlrRegression()
