# _*_ coding:utf-8 _*_  //为了告诉python解释器，按照utf-8编码读取源代码，否则，你在源代码中写的中文输出可能会由乱码  

from numpy import * 

#--------------------------Logistic回归：权重优化，步长，迭代次数，错误率-------------------------# 
def loadDataSet():#打开文件逐行读取
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])#假设初始label为
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):#梯度上升法
    dataMatrix = mat(dataMatIn)             # convert to NumPy matrix
    labelMat = mat(classLabels).transpose() # convert to NumPy matrix && 转置
    m,n = shape(dataMatrix) # 100 * 3 
    alpha = 0.001 # 移动步长
    maxCycles = 500 # 迭代次数
    weights = ones((n,1)) # 每个特征的权重 X0,X1,X2
    for k in range(maxCycles):                # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)     # matrix mult 100*1
        error = (labelMat - h)                # vector subtraction 按照该差值的方向调整回归系数 100*1
        weights = weights + alpha * dataMatrix.transpose() * error #matrix mult 3*1 ？？？
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()#画图
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):#随机梯度上升法
    m,n = shape(dataMatrix)#数组
    alpha = 0.01#迭代次数为1
    weights = ones(n)   #initialize to all ones
    for i in range(m):#dataMatrix[i]，classLabels[i]——每次只用一个样本点
        h = sigmoid(sum(dataMatrix[i]*weights))#h为数值，不是矩阵
        error = classLabels[i] - h#error为数值，不是矩阵
        weights = weights + alpha * error * dataMatrix[i]#数值
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter):#随机梯度上升法的改进
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))#对应2
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #1.alpha 每次迭代时不断调整
            randIndex = int(random.uniform(0,len(dataIndex)))#2.随机选取更新样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])#3.删除已经使用过的样本
    return weights
#--------------------------------------------完毕--------------------------------------------#    


#------------------------------------疝气病预测马的死亡率---------------------------------------#
def classifyVector(inX, weights):#输入：一行样本，权重，输出：分类值
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []#训练
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):#0-20
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)#训练

    errorCount = 0; numTestVec = 0.0#测试
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):#测试
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))















