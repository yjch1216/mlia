# -*- coding: utf-8 -*-
from numpy import *
from numpy import linalg as la


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
#相似度1：欧式距离
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))
#相似度2：威尔逊距离
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]
#相似度3：余弦
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# 遍历 对于每一个user没有评价过的item，假设用户A1和user都评价过商品j1，同时用户A1还评价过商品item，这样我们就根据
# 用户A1对item的评价得到用户user对item的评价。1.假设A1和A2。。。都评价过商品j，则一起来计算相似度，然后得到评价。
# 2.假设用户B1和user都评价过商品j2,同事评价了商品item，这是要根据j1和j2两次综合，计算用户user对item的评价。
# 这就是协同过滤，将用户和其他用户对比，根据商品相似度，得到评价。
def standEst(dataMat, user, simMeas, item):#数据矩阵、用户编号、相似度计算方法和物品编号
    n = shape(dataMat)[1]
    simTotal = 0.0;ratSimTotal = 0.0
    for j in range(n):#对所有user评价过的那个商品，
        userRating = dataMat[user, j]
        if userRating == 0: continue
        # overLap为假设用户A1和user都评价过商品j，同时还评价过商品item的用户Ai
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:# 根据用户overLap 们 计算商品 j 和 item 的相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity # 计算所有评价产品累计相似度
        ratSimTotal += similarity * userRating  #根据评分计算比率
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

#利用SVD
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0;ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat) #不同于stanEst函数，加入了SVD分解
    Sig4 = mat(eye(4) * Sigma[:4])  # 建立对角矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I #降维：变换到低维空间
    #下面依然是计算相似度，给出归一化评分
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1] #寻找用户未评价的产品
    if len(unratedItems) == 0: return ('you rated everything')
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)#基于相似度的评分
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


#实例：SVD实现图像压缩

#打印矩阵。由于矩阵包含了浮点数,因此必须定义浅色和深色。
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print (1,)
            else: print (0,)
        print ('')

#压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print ("****original matrix******")
    #printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat) #SVD分解
    SigRecon = mat(zeros((numSV, numSV))) #创建初始特征
    for k in range(numSV):#构造对角矩阵
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print ("****reconstructed matrix using %d singular values******" % numSV)
    #printMat(reconMat, thresh)