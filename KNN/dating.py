import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import operator
"""
函数说明：打开并解析文件，对数据进行分类，1代表不喜欢，2代表魅力一般，3代表极具魅力
Parameters:
    filename-文件名
Return:
    returnMat-特征矩阵
    classLabelVector-分类label向量
Modify
    2018-7-26
"""
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    array0Lines = fr.readlines()
    #得到文件行数
    numberOfLines = len(array0Lines)
    #返回的numpy矩阵，解析完成的数据：numberOfLines行，3列
    returnMat = np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in array0Lines:
        #s.strip(rm),当rm为空时，默认删除空白符（包括'\n','\r','\t','')
        line = line.strip()
        #使用s.split(str="",num = string, cout(str))，将字符串根据'\t'分隔符进行切片
        listFromLine = line.split('\t')
        #将数据前三列提取出来，存放到returnMat中，即特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类，1代表不喜欢，2代表魅力一般，3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

"""
函数说明：对数据进行归一化
Parameters:
    dataSet-特征矩阵
Return:
    normDataaSet-归一化后的特征矩阵
    ranges-数据范围
    minVals-数据最小值
Modify
    2018-7-28
"""
def autoNorm(dataSet):
    #获得每一列数据的最小，最大值，返回行
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSetd的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet-np.tile(minVals,(m,1))
    #除以最大最小值的差，得到归一化数据
    normDataSet /= np.tile(ranges,(m,1))
    #返回归一化数据，数据范围，最小值
    return normDataSet,ranges,minVals
def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX，在行向量方向上重复inXdataSetSize次
    diffMat = np.tile(inX,(dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis = 1)
    #开方，计算出距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key, default = None),字典的get()方法，返回指定键的值，如果值不在字典中返回默认值
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #python3中用items()
    #key = operator.itemgetter(0)根据字典的值进行排序
    #key = operator.itemgetter(1)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    #返回次数最多的类，即所要分类的类别
    return sortedClassCount[0][0]

"""
函数说明：可视化数据
Parameters:
    datingDataMat-特征矩阵
    datingLabels-分类label
Return:
    无
Modify
    2018-7-27
"""
def showData(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname = r"c:\windows\fonts\simsun.ttc",size=14)
    #将fig画布分隔成rows行cols列，不共享x和y轴，fig画布的大小为（13,8）
    #当nrows = 2，ncols = 2时，代表fig画布被分成四个区域，axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))
    numberOfLabels = len(datingLabels)
    labelsColors = []
    for i in datingLabels:
        if i == 1:
            labelsColors.append('black')
        if i == 2:
            labelsColors.append('orange')
        if i == 3:
            labelsColors.append('red')
    #画出散点图，以datingDataMat矩阵的第一（飞行常客里程），第二列（玩游戏）数据画散点数据，散点大小为15,透明度为0.5
    axs[0][0].scatter(x = datingDataMat[:,0],y=datingDataMat[:,1],color=labelsColors,s=15,alpha =.5)
    #设置标题，x轴label，y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties = font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties = font)
    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs0_ylabel_text,size=7,weight='bold',color='black')
    # 画出散点图，以datingDataMat矩阵的第一（飞行常客里程），第三列（冰激凌）数据画散点数据，散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2],color = labelsColors, s = 15, alpha = .5)
    # 设置标题，x轴label，y轴label
    axs0_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs0_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties = font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    # 画出散点图，以datingDataMat矩阵的第而（玩游戏），第三列（冰激凌）数据画散点数据，散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2],color = labelsColors, s = 15, alpha = .5)
    # 设置标题，x轴label，y轴label
    axs0_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs0_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
    didntLike = mlines.Line2D([],[],color = 'black',marker='.',markersize=6,label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='samllDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    #显示图片
    plt.show()
    """
函数说明：分类器测试函数
Parameters:
    无
Return:
    无
Modify
    2018-7-28
"""
def datingClassTest():
    # 打开的文件名
    filename = "datingTestSet.txt"
    #返回特征矩阵和分类向量
    datingDataMat, datingLabels = file2matrix(filename)
    #取所有数据的百分之十
    hoRatio = 0.05
    #数据归一化，返回归一化后的矩阵，数据范围，数据最小值
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    #获得normDataSet的行数
    m = normDataSet.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m*hoRatio)
    #分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],10)
        print("分类结果：%d\t真实类别：%d"%(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率：%f%%"%(errorCount/float(numTestVecs)*100))

    """
函数说明：
Parameters:
    无
Return:
    无
Modify
    2018-7-26
"""
def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所消耗时间比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每年消费的冰激凌公升数："))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 返回特征矩阵和分类向量
    datingDataMat, datingLabels = file2matrix(filename)
    # 数据归一化，返回归一化后的矩阵，数据范围，数据最小值
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    #生成测试集
    inArr = np.array([ffMiles,precentTats,iceCream])
    #测试数据集归一化
    norminArr = (inArr - minVals)/ranges
    #返回分类结果
    classifierResult = classify0(norminArr,normDataSet,datingLabels,3)
    #打印结果
    print("你可能%s这个人"%(resultList[classifierResult-1]))
    """
函数说明：main
Parameters:
    无
Return:
    无
Modify
    2018-7-26
"""
if __name__ == '__main__':
    classifyPerson()