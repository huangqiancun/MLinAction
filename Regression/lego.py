from bs4 import BeautifulSoup
import numpy as np
import random

def scrapePage(retX,retY,inFile,yr,numPce,origPre):
    """
    从页面读取数据，生成retX和retY列表
    :param retX: 数据X
    :param resY: 数据Y
    :param inFile: HTML文件
    :param yr: 年份
    :param numPce: 乐高部件数目
    :param origPre: 原价
    :return: 无
    """
    #打开并读取HTML文件
    with open(inFile,encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    #根据HTML页面结构进行解析
    currentRow = soup.find_all('table',r='%d'%i)
    while len(currentRow)!=0:
        currentRow = soup.find_all('table', r='%d' % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        #查找是否有全新标签
        if (lwrTitle.find('new')>-1)or(lwrTitle.find('nisb')>-1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        #查找是否已经标志售出，我们只收集已经售出的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品#%d没有售出"%i)
        else:
            #解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping','')
            sellingPrice = float(priceStr)
            #去掉不完整的套装价格
            if sellingPrice>origPre*0.5:
                print("%d\t%d\t%d\t%f\t%f"%(yr,numPce,newFlag,origPre,sellingPrice))
                retX.append([yr,numPce,newFlag,origPre])
                retY.append(sellingPrice)
        i+=1
        currentRow = soup.find_all('table',r="%d"%i)

def setDataCollect(retX,retY):
    """
    依次读取六种乐高套装的数据，并生成矩阵
    :param retX:
    :param retY:
    :return:
    """
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99

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
    print(inMeans)
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
def ridgeRegres(xMat,yMat,lam =0.2):
    """
    领回归
    :param
        xMat:x数据集
        yMat:y数据集
        lam:缩减系数
    :return:
        ws：回归系数
    """
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵，不能转置")
        return
    ws = denom.I*(xMat.T*yMat)
    return ws

def useStandRegres():
    """
    使用简单的线性回归
    :return: 无
    """
    lgX=[]
    lgY=[]
    setDataCollect(lgX,lgY)
    data_num,features_num = np.shape(lgX)
    lgX1 = np.mat(np.ones((data_num,features_num+1)))
    lgX1[:,1:5] = np.mat(lgX)
    ws = standRegression(lgX1,lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (ws[0], ws[1], ws[2], ws[3], ws[4]))

def crossValidation(xArr,yArr,numVal = 10):
    """
    交叉验证岭回归
    :param xArr:x数据集
    :param yArr: y数据集
    :param numVal: 交叉验证次数
    :return: 回归系数矩阵
    """
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal,30))
    for i in range(numVal):
        trainX = [];trainY = []
        testX=[];testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX = np.mat(testX);matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain
            yEst = matTestX*np.mat(wMat[k,:]).T+np.mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A,np.array(testY))
    meanErrors = np.mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    xMat = np.mat(xArr);yMat = np.mat(yArr).T
    meanX= np.mean(xMat,0);varX = np.var(xMat,0)
    unReg = bestWeights/varX
    print("%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价"%((-1*np.sum(np.multiply(meanX,unReg))+np.mean(yMat)),unReg[0,0],unReg[0,1],unReg[0,2],unReg[0,3]))

def ridgeTest(xArr,yArr):
    """
    领回归测试
    :param
        xArr:x数据集
        yArr:y数据集
    :return:
        wMat:回归系数矩阵
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    #数据标准化
    yMean = np.mean(yMat,axis=0)#行与行操作，求均值
    yMat = yMat-yMean#数据减去均值
    xMeans = np.mean(xMat,axis = 0)#行与行操作，求均值
    xVar = np.var(xMat,axis = 0)#行与行操作，求方差
    xMat = (xMat-xMeans)/xVar#数据减去均值除以方差实现标准化
    numTestPts = 30#30个不同的lambda测试
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))#初始回归系数矩阵
    for i in range(numTestPts):#改变lambda计算回归系数
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))#lambda以e的指数变化，最初是一个非常小的数
        wMat[i,:] = ws.T#计算回归系数矩阵
    return wMat
if __name__=='__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX,lgY)
    print(ridgeTest(lgX,lgY))
