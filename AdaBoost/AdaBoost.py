import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    """
    创建单层决策树的数据集
    :param：
        无
    :return:
        dataMat-数据矩阵
        classLabels-数据标签
    """
    dataMat = np.mat([[1.,2.1],
                      [1.5,1.6],
                      [1.3,1.],
                      [1.,1.],
                      [2.,1.]])
    classLabels = [1.,1.,-1.,-1.,1.]
    return dataMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal,threshIneq):
    """
    单层决策树分类函数
    :param dataMatrix:数据矩阵
    :param dimen: 第dimen列，也就是第几个特征
    :param threshVal: 阈值
    :param threshIneq: 标志
    :return retArray：分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0],1))#初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal] = -1.0#如果小于阈值，则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0#如果大于阈值，则赋值为-1
    return retArray

def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    :param dataArr: 数据矩阵
    :param classLabels: 数据标签
    :param D:样本权重
    :return bestStump：最佳单层决策树
    :return minError：最小误差
    :return bestClasEst：最佳的分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')#最小误差初始化为无穷
    for i in range(n):#变量所有特征
        rangeMin = dataMatrix[:,i].min()#找到特征的最小值
        rangeMax = dataMatrix[:,i].max()#找到特征的最大值
        stepSize = (rangeMax-rangeMin)/numSteps#计算步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:#大于和小于的都变量，lt：less than，gt：greater than
                threshVal = (rangeMin+float(j)*stepSize)#计算阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#计算分类结果
                errArr = np.mat(np.ones((m,1)))#初始化误差矩阵
                errArr[predictedVals == labelMat] =0#分类正确的，赋值为0
                weightedError = D.T*errArr#计算误差
                print("split:dim%d,thresh %.2f,thresh ineqal:%s,the weighted error is %.3f"%(i,threshVal,inequal,weightedError))
                if weightedError < minError:#找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt = 20):
    """
    adaboost训练决策树
    :param dataArr: 数据矩阵
    :param classLabels: 数据标签
    :param numIt: 迭代次数
    :return: 分类结果
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)#初始化权重
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#构建单层决策树
        print("D:",D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#计算弱分类器的权重，使error不等于0，因为分母不能为0
        bestStump['alpha'] = alpha#存储弱分类器权重
        weakClassArr.append(bestStump)#存储单层决策树
        print("classEst:",classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)#计算e的指数项
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()#根据权重计算公式，更新权重
        #计算adaboost误差，当误差为0的时候，退出循环
        aggClassEst += alpha*classEst
        print("aggClassEst:",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))#计算误差
        errorRate = aggErrors.sum()/m
        print("total error:",errorRate)
        if errorRate ==0.0:#误差为0，退出
            break
    return weakClassArr,aggClassEst

def adaClassify(dataToClass,classifierArr):
    """
    adaboost分类函数
    :param dataToClass:待分类样例
    :param classifierArr: 训练好的分类器
    :return: 分类结果
    """
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):#变量所有分类器，进行分类
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def showDataSet(dataMat,labelMat):
    """
    数据可视化
    :param dataMat:数据矩阵
    :param labelMat: 数据标签
    :return: 无
    """
    data_plus = []#正样本
    data_minus = []#负样本
    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)#转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1])#正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1])#负样本散点图
    plt.show()

if __name__=='__main__':
    dataArr,classLabels = loadSimpData()
    weakClassArr,aggClassEst = adaBoostTrainDS(dataArr,classLabels)
    print(adaClassify([[0,0],[5,5]],weakClassArr))
