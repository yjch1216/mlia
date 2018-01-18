# _*_ coding:utf-8 _*_  //为了告诉python解释器，按照utf-8编码读取源代码，否则，你在源代码中写的中文输出可能会由乱码  

from numpy import *

def loadDataSet(fileName):      #读取一个以tab为分隔符的文件
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #将每行映射成float
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):#根据feature列的value的值将dataSet切分成切分成两个子集
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]#数组过滤
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

def regLeaf(dataSet):#建立叶节点
    return mean(dataSet[:,-1])#目标变量均值

def regErr(dataSet):#误差估计
    return var(dataSet[:,-1]) * shape(dataSet)[0]#var求得的是方差，*m就是方差的和了

def linearSolve(dataSet):   #格式化Y和X
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#X的第一列为1，第二列为dataSet的第0列
    xTx = X.T*X #线性回归，同前
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):# 建立叶节点模型
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet): # 计算误差，用于找到最佳切分点
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#CART方法
    #用最佳方式切分数据集，生成相应的叶节点，遍历所有的特征及其可能的取值找到使得误差最小的切分阈值
    #leafType 给出建立叶节点的函数； errType 为误差计算函数 ；ops为所需的其他参数的一个元组 
    tolS = ops[0]; tolN = ops[1]#控制函数停止时机：分别为误差下降值和切分的最小样本数，对参数敏感
    # tolS越大，分叉越少 ， tolN越大，分叉越少
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #剩余特征数目为1，直接返回（预剪枝）
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):#对每个特征
        for splitVal in set(dataSet[:,featIndex]):#对每个特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)#将数据切分成两份
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)#计算切分误差
            if newS < bestS: #当前误差小于当前记录的最小误差
                bestIndex = featIndex#最佳切分特征更新
                bestValue = splitVal#最佳切分特征对应的特征值更新
                bestS = newS#更新最小误差
    if (S - bestS) < tolS: #切分后效果提升不大
        return None, leafType(dataSet) #不切分，直接创建叶节点
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #如果某个子集的大小小于用户指定的数目
        return None, leafType(dataSet) #不切分，直接创建叶节点
    return bestIndex,bestValue#返回得到的切分特征和特征值

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat
#leafType 给出建立叶节点的函数； errType 为误差计算函数 ；ops为所需的其他参数的一个元组，
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#切分树的函数
    if feat == None: return val #满足停止条件，不能再分，返回叶节点，回归树模型是一个常数，模型树则模型是一个线性方程
    retTree = {}#不满足停止条件就继续分
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)#递归调用
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):#用于测试obj是否为一棵树
    return (type(obj).__name__=='dict')

def getMean(tree):#从上到下遍历直到叶节点
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):#从上到下找到叶节点，用测试集判断叶节点合并是否能降低测试误差，后剪枝
    if shape(testData)[0] == 0: return getMean(tree) #测试集为空，求均值
    if (isTree(tree['right']) or isTree(tree['left'])):#左右均为树，则切分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)#是子树，则继续剪纸
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #左右均为叶节点，进行合并？
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])#先根据树划分测试集
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))#合并前误差
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))#合并后误差
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean#叶节点合并
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):#树回归
    return float(model)

def modelTreeEval(model, inDat):#模型树
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):#遍历树，直到叶节点
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):#返回预测值y
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat