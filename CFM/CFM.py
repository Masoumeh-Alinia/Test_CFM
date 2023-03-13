
import tensorflow as tf
import numpy as np
# import pandas as pd
from .BMF import  BMF
import time
from collections import deque
from .regression import regressionModel
# in RMF class, the thrid columns is the global ratings.

class CFM(BMF):
    def __init__(self,mod = 'Linear',criteriaNum = 3,lambd = 0.01,alpha = 0.01,regressionReg=0.01, **kwargs):
        #criteriaNum mean the Number of sub-criteria/subrating
        super(CFM, self).__init__(**kwargs)
        self.criteriaNum = criteriaNum
        self.lambd = lambd
        self.alpha = alpha
        self.subRatingBatch = tf.placeholder(tf.float32, [None,self.criteriaNum], name='ratingBatch')
        self.userSubWeight = tf.Variable(tf.random_normal([self.users, self.K,self.criteriaNum],stddev = 0.1)/ np.sqrt(self.K))
        self.itemSubWeight = tf.Variable(tf.random_normal([self.items, self.K,self.criteriaNum],stddev = 0.1)/ np.sqrt(self.K))
        self.userSubBias = tf.Variable(tf.zeros([self.users,self.criteriaNum]))
        self.itemSubBias = tf.Variable(tf.zeros([self.items,self.criteriaNum]))
        self.subRatingBatch = tf.placeholder(tf.float32, [None,criteriaNum], name='subRatingBatch')
        self.regression = regressionModel(_mod = mod,reg=regressionReg)


    def getData(self, train=None, test=None, ignore=False,):
        m = train.max() + 1
        if (not ignore) and (m[0] != self.users):
            self.logger.info(
                'error: the user num [%d] is not equal the maximum user ID [%d] in data,please check or setting ignore=True' % (
                    self.users, m[0]))
            #return
        if (not ignore) and (m[1] != self.items):
            self.logger.info(
                'error: the item num [%d] is not equal the maximum item ID [%d] in data,please check or setting ignore=True' % (
                    self.items, m[1]))
            #return
        self.trainData = train.values[:, :3 + self.criteriaNum ]
        self.testData = test.values[:, :3 ]
        self.trRow = self.trainData.shape[0]
        self.tsRow = self.testData.shape[0]

    def buildGraph(self):
        # userBatch, itemBatch, ratingBatch = it.get_next()
        self.globalBias = tf.constant(self.trainData[:,2:3].mean(axis=0),dtype=tf.float32)
        self.globalSubBias = tf.constant(self.trainData[:,3:].mean(axis=0),dtype=tf.float32)
        userWeightBatch = tf.nn.embedding_lookup(self.userWeight, self.userBatch)
        itemWeightBatch = tf.nn.embedding_lookup(self.itemWeight, self.itemBatch)

        userBiasBatch = tf.nn.embedding_lookup(self.userBias, self.userBatch)
        itemBiasBatch = tf.nn.embedding_lookup(self.itemBias, self.itemBatch)

        output = tf.reduce_sum(tf.multiply(userWeightBatch, itemWeightBatch), 1)
        output = tf.add(output, self.globalBias)
        output = tf.add(output, userBiasBatch)
        globalPrediction = tf.add(output, itemBiasBatch)

        userSubWeightBatch = tf.nn.embedding_lookup(self.userSubWeight, self.userBatch)
        itemSubWeightBatch = tf.nn.embedding_lookup(self.itemSubWeight, self.itemBatch)

        userSubBiasBatch = tf.nn.embedding_lookup(self.userSubBias, self.userBatch)
        itemSubBiasBatch = tf.nn.embedding_lookup(self.itemSubBias, self.itemBatch)

        subOutput = tf.reduce_sum(tf.multiply(userSubWeightBatch, itemSubWeightBatch), 1)
        subOutput = tf.add(subOutput, self.globalSubBias)
        subOutput = tf.add(subOutput, userSubBiasBatch)
        subPrediction = tf.add(subOutput, itemSubBiasBatch)
        subCost = self.lambd *tf.nn.l2_loss(tf.subtract(subPrediction, self.subRatingBatch))

        subPrediction = tf.clip_by_value(subPrediction, self.minRating, self.maxRating)
        prediction = (1-self.alpha)*globalPrediction +  self.alpha * tf.reshape(self.regression(subPrediction),[-1])
        base = tf.nn.l2_loss(tf.subtract(prediction, self.ratingBatch))



        subReg = tf.add(self.uReg *tf.nn.l2_loss(userSubWeightBatch), self.iReg * tf.nn.l2_loss(itemSubWeightBatch))
        globalReg = tf.add(self.uReg *tf.nn.l2_loss(userWeightBatch), self.iReg * tf.nn.l2_loss(itemWeightBatch))
        reg = tf.add(subReg, globalReg)
        bReg = tf.add(tf.nn.l2_loss(userBiasBatch), tf.nn.l2_loss(itemBiasBatch))
        reg = tf.add(reg, self.biasReg *bReg)

        subBReg  = tf.add(tf.nn.l2_loss(userSubBiasBatch), tf.nn.l2_loss(itemSubBiasBatch))
        reg = tf.add(reg,  self.biasReg *subBReg) + tf.losses.get_regularization_loss()
        globalCost = tf.add(base, reg)

        cost = globalCost + subCost

        return prediction, cost

    def testIter(self,pre):
        num_batch_loop = int(self.tsRow / self.tsbatchSize)
        prediction = []
        errors = deque()
        t1 = time.time()
        for i in range(num_batch_loop):
            pred_batch = pre.eval(
                {self.userBatch: self.testData[i * self.tsbatchSize:(i + 1) * self.tsbatchSize, 0],
                 self.itemBatch: self.testData[i * self.tsbatchSize:(i + 1) * self.tsbatchSize, 1]},session=self.sess)
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)
            prediction.append(pred_batch)
            errors.append(np.mean(
                    np.power(pred_batch - self.testData[i * self.tsbatchSize:(i + 1) * self.tsbatchSize, 2], 2)))
        if (self.tsRow % self.tsbatchSize):
            pred_batch = pre.eval(
                {self.userBatch: self.testData[(i + 1) * self.tsbatchSize:, 0],
                 self.itemBatch: self.testData[(i + 1) * self.tsbatchSize:, 1]}, session=self.sess)
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)
            prediction.append(pred_batch)
            errors.append(np.mean(
                    np.power(pred_batch - self.testData[(i + 1) * self.tsbatchSize:, 2], 2)))

            #TS_epoch_loss = np.sqrt(np.mean(errors))
                #RMSEts.append(TS_epoch_loss)

                #self.testPrediction = np.concatenate(prediction)
        #tsRMSE = np.sqrt(np.mean(errors))
        err = np.concatenate(prediction) - self.testData[:, 2]
        tsRMSE = np.sqrt(np.power(err, 2).mean())
        tsMAE = np.abs(err).mean()
        if self.debug:
            with open('debug.txt','a+') as f:
                f.write(','.join(map(str,err)))
                f.write('\n')
        return tsMAE,tsRMSE,time.time() - t1,prediction
 
    
    def trainIter(self, pre, cost, optimizer,shuffle):
        errors = deque()
        num_batch_loop = int(self.trRow / self.batchSize)
        if shuffle:
            np.random.shuffle(self.trainData)
        prediction = np.asarray([])
        t = time.time()
        for i in range(num_batch_loop):
            _, c, pred_batch = self.sess.run([optimizer, cost, pre],
                                             feed_dict={self.userBatch: self.trainData[
                                                                        i * self.batchSize:(i + 1) * self.batchSize, 0],
                                                        self.itemBatch: self.trainData[
                                                                        i * self.batchSize:(i + 1) * self.batchSize, 1],
                                                        self.ratingBatch: self.trainData[
                                                                          i * self.batchSize:(i + 1) * self.batchSize,
                                                                          2:3],
                                                        self.subRatingBatch: self.trainData[i * self.batchSize:(
                                                                                                                           i + 1) * self.batchSize,
                                                                             3:]})
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)

            # print type(pred_batch)
            errors.append(np.mean(
                np.power(pred_batch - self.trainData[i * self.batchSize:(i + 1) * self.batchSize, 2], 2)))
        if self.trRow % self.batchSize:
            _, c, pred_batch = self.sess.run([optimizer, cost, pre],
                                             feed_dict={self.userBatch: self.trainData[(i + 1) * self.batchSize:, 0],
                                                        self.itemBatch: self.trainData[(i + 1) * self.batchSize:, 1],
                                                        self.ratingBatch: self.trainData[(i + 1) * self.batchSize:,
                                                                          2:3],
                                                        self.subRatingBatch: self.trainData[(i + 1) * self.batchSize:,
                                                                             3:]})
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)
            # print type(pred_batch)
            errors.append(np.mean(
                np.power(pred_batch - self.trainData[(i + 1) * self.batchSize:, 2], 2)))

            # self.trainPrediction = prediction
            # pred_batch = np.clip(pred_batch, self.minRating, self.maxRating)
            # ttt = time.time()
        trRMSE = np.sqrt(np.mean(errors))
        return trRMSE, time.time() - t

    def train(self, mod = 'CPU',num_CPU = 1, num_GPU = 1,epochs = None,shuffle = True):
        if epochs ==None:
            epochs = self.epochs
        self.RMSEtr = []
        self.RMSEts = []
        # bestRmse  = 100
        self.wHis = []
        # self.sess.run(merged)
        pre, cost, optimizer = self.creatSess(mod=mod, num_CPU=num_CPU,num_GPU = num_GPU )
        self.logger.info('users = %d, items = %d, K = %d, u_reg = %.1e, i_reg = %.1e,  b_reg = %.1e, learningRate = %.1e, batchSize = %d, epochs = %d,  minRating = %d, maxRating = %d,optimizer = %s'
              %(self.users, self.items, self.K, self.uReg, self.iReg, self.biasReg , self.learningRate, self.batchSize, self.epochs, self.minRating, self.maxRating, optimizer.name))
        tsMAE,tsRMSE,testTime,prediction = self.testIter(pre)
        #RMSEts.append(tsRMSE)
        #precisions,recalls,ndcgs
        bestRmse = tsRMSE
        self.testPrediction = np.concatenate(prediction)
        self.logger.info( "Init Test loss:" + str(round(tsRMSE, 4)) + ' Test time: ' + str(round(testTime, 3)))

        for epoch in range(epochs):
            self.wHis.append(self.sess.run(self.regression.weights[0]))
            #writer.add_summary(summary,epoch)
            trRMSE,trainTime = self.trainIter(pre, cost, optimizer,shuffle)
            self.RMSEtr.append(trRMSE)
            tsMAE,tsRMSE, testTime, prediction = self.testIter(pre)
            self.RMSEts.append(tsRMSE)
            precisions,recalls,ndcgs = self.rankTest(pre)
            
            if tsRMSE<bestRmse:
                bestRmse = tsRMSE
                self.testPrediction = np.concatenate(prediction)
                self.save(bestRmse,tsMAE)
            self.logger.info("Epoch " + str(epoch + 1) + " completed out of " + str(self.epochs) 
                             + "; Train loss:" + str(round(trRMSE, 4)) 
                             + ' Train time: ' + str(round(trainTime, 3))
                             + "; Test loss:" + str(round(tsRMSE, 4))  
                             + ", Test MAE:" + str(round(tsMAE, 4))
                             + ', precsion: %.4f, recall: %.4f, NDCG: %.4f '%(np.mean(precisions),np.mean(recalls),np.mean(ndcgs)) 
                             + ' Test time: ' + str(round(testTime, 3)))
        #print Final info
        err= self.testPrediction - self.testData[:,2]
        self.logger.info('best RMSE : %s'% str(round(np.sqrt(np.power(err, 2).mean()), 5)))
        self.logger.info('best MAE : %s'% str(round(np.abs(err).mean(), 5)))

