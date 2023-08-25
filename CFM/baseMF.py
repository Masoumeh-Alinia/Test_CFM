import tensorflow as tf
import numpy as np
#import pandas as pd
import time
import logging
# from remtime import *
from collections import deque
import sys
import heapq # for retrieval topK
import math

class baseMF(object):
    def __init__(self, K=5, reg = None, uReg=0.01, iReg=0.01, learningRate=0.005, batchSize=512, epochs=20, users=6041, items=3953,
                 minRating=1., maxRating=5.,pretainName = None, optimizer = 'adam',tsbatchSize=500,th=.89,saveStr ='',log=None,debug = False):
        self.K = K
        #self.reg = reg
        if reg:
            self.reg = reg
            self.iReg = reg
            self.uReg = reg
        else:
            self.iReg = iReg
            self.uReg = uReg
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.tsbatchSize = tsbatchSize
        self.epochs = epochs
        self.users = users
        self.items = items
        self.minRating = minRating
        self.maxRating = maxRating
        self.userBatch = tf.compat.v1.placeholder(tf.int32, [None], name='userBatch')
        self.itemBatch = tf.compat.v1.placeholder(tf.int32, [None], name='itemBatch')
        self.ratingBatch = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ratingBatch')
        tf.compat.v1.disable_eager_execution()
        #self.userWeight = tf.Variable(tf.random_normal([self.users, self.K]) / np.sqrt(self.users))
        #self.itemWeight = tf.Variable(tf.random_normal([self.items, self.K]) / np.sqrt(self.items))
        self.userWeight = tf.Variable(tf.random_normal([self.users, self.K],stddev = 0.1) / np.sqrt(self.K))
        self.itemWeight = tf.Variable(tf.random_normal([self.items, self.K],stddev = 0.1)/ np.sqrt(self.K) )
        #self.userWeight = tf.Variable(tf.random_normal([self.users, self.K]))
        #self.itemWeight = tf.Variable(tf.random_normal([self.items, self.K]))
        self.saveStr = saveStr
        self.th =th
        self.debug = debug

        self.init = False
        self.pretainName = pretainName
        self.optimizer = optimizer
        if log!=None:
            self.logerConfig(log.replace('.log','_%s.log'%self.__class__.__name__))

    def logerConfig(self,log):
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.INFO)
        #
        fh = logging.FileHandler(log)
        fh.setLevel(logging.INFO)
        #
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        #
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        #
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


    def getData(self, train=None, test=None, ignore=False):

        m = train.max()+ 1
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
        if self.__class__.__name__ == 'SVDPP':
            #print(self.__class__.__name__)
            self.trainData = train.sort_values([0, 1]).values[:, :3]  # .repeat(self.epochs)
            self.testData = test.sort_values([0, 1]).values[:, :3]
        else:
            self.trainData = train.values[:, :3]
            self.testData = test.values[:, :3]

        self.trRow = self.trainData.shape[0]
        self.tsRow = self.testData.shape[0]



    def buildGraph(self):
        # userBatch, itemBatch, ratingBatch = it.get_next()
        userWeightBatch = tf.nn.embedding_lookup(self.userWeight, self.userBatch)
        itemWeightBatch = tf.nn.embedding_lookup(self.itemWeight, self.itemBatch)
        prediction = tf.reduce_sum(tf.multiply(userWeightBatch, itemWeightBatch), 1)
        base = tf.nn.l2_loss(tf.subtract(prediction, self.ratingBatch))
        #reg = self.reg * tf.add(tf.nn.l2_loss(userWeightBatch), tf.nn.l2_loss(itemWeightBatch))
        reg =  tf.add(self.uReg *tf.nn.l2_loss(self.userWeight), self.iReg * tf.nn.l2_loss(self.itemWeight))
        cost = tf.add(base, reg)
        # r = tf.clip_by_value(prediction,self.minRating,self.maxRating)
        # error = tf.reduce_mean(tf.pow(tf.subtract(r, self.ratingBatch),2))
        # print ('OK')
        return prediction, cost  # ,error

    def creatSess(self,mod = 'CPU', num_CPU = 1, num_GPU = 0):
        '''
        if mod.lower() =='cpu':
            num_CPU = cpu_cores#cpu_cores
        elif mod.lower() == 'gpu':
            num_GPU = num_GPU
        elif mod.lower() =='auto':
            if self.K >99:
                num_GPU = 1
        '''
        config = tf.ConfigProto(intra_op_parallelism_threads=num_CPU,
                                inter_op_parallelism_threads=num_CPU*2,
                                allow_soft_placement=True,
                                device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
        if num_GPU > 0:
            config.gpu_options.allow_growth = True
        #session = tf.Session(config=config)
        #print('1aa----------------',num_CPU,num_GPU)
        #num_batch_loop = int(NUM_TR_ROW / BATCH_SIZE)
        pre, cost = self.buildGraph()
        # tsPre,_,tsError = self.buildGraph(it=self.testIter)
        if self.optimizer.lower() =='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(cost)
        elif self.optimizer.lower() =='rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learningRate).minimize(cost)
        elif self.optimizer.lower() =='momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learningRate).minimize(cost)
        elif self.optimizer.lower() =='adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learningRate).minimize(cost)
        elif self.optimizer.lower() =='adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learningRate).minimize(cost)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(cost)

        #saver = tf.train.Saver()

        #with tf.Session() as sess:
        self.saver = tf.train.Saver(tf.global_variables())
        #if not(self.pretainName == None):
            #self.redPretrain()
        #    self.init = True
        if not self.init:
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.init = True
        return pre, cost,optimizer


    def restore(self,modelName):
        self.creatSess()
        #saver = tf.train.Saver()
        self.saver.restore(self.sess, modelName)

    def save(self,tsRMSE,tsMAE):
        if tsRMSE < self.th:
            self.saver.save(self.sess, 'Model/%s_%s_RMSE_%.4f_MAE_%.4f' % (self.saveStr,self.__class__.__name__, tsRMSE,tsMAE))
    
    def testIter(self,pre):
        num_batch_loop = int(self.tsRow / self.tsbatchSize)
        prediction = []
        #errors = deque()
        t1 = time.time()
        for i in range(num_batch_loop):
            pred_batch = pre.eval(
                {self.userBatch: self.testData[i * self.tsbatchSize:(i + 1) * self.tsbatchSize, 0],
                 self.itemBatch: self.testData[i * self.tsbatchSize:(i + 1) * self.tsbatchSize, 1]}, session=self.sess)
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)
            prediction.append(pred_batch)
            #errors.append(np.mean(
            #    np.power(pred_batch - self.testData[i * self.tsbatchSize:(i + 1) * self.tsbatchSize, 2], 2)))
        if self.tsRow % self.tsbatchSize:
            pred_batch = pre.eval(
                {self.userBatch: self.testData[(i + 1) * self.tsbatchSize:, 0],
                 self.itemBatch: self.testData[(i + 1) * self.tsbatchSize:, 1]}, session=self.sess)
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)
            prediction.append(pred_batch)
        tsRMSE = np.sqrt(np.power(np.concatenate(prediction) - self.testData[:, 2], 2).mean())
        tsMAE = np.abs(np.concatenate(prediction) - self.testData[:, 2]).mean()
        return tsMAE,tsRMSE,time.time() - t1,prediction

    def rankTest(self,pre,_K=10,onlyHigh = True,mod=False):
        if not mod:
            return [0],[0],[0]
        recall =[]
        ndcg = []
        precision= []
        testData = self.testData
        if onlyHigh : 
            high = np.max(self.trainData[:,2])
            testData = testData[testData[:,2]==high]
        for u in np.unique(testData[:,0]):
            pred_batch = pre.eval(
                    {self.userBatch: [u]*self.items,
                     self.itemBatch: range(self.items)},session=self.sess)
            map_item_score=dict(zip(range(self.items),pred_batch))
            trainList = self.trainData[self.trainData[:,0]==u][:,1].astype(np.int32)
            pred_batch[trainList] = -np.inf
            ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
            f = testData[testData[:,0]==u][:,1].astype(np.int32)
            idx = 0
            nn = 0
            ndcf= 0
            for i in ranklist:
                #ndc_f = []
                if  i in f:
                    nn = nn+1
                    ndcf = ndcf + math.log(2)/1. / math.log(idx+2)
                idx = idx+1
            precision.append(nn/1./_K)
            recall.append(nn/1./len(f))
            ndcg.append(ndcf/len(f))
        return precision,recall,ndcg

    def trainIter(self, pre, cost, optimizer,shuffle):
        num_batch_loop = int(self.trRow / self.batchSize)
        # sess.run([self.trainIter.initializer,self.testIter.initializer])
        if shuffle:
            np.random.shuffle(self.trainData)
        prediction = []
        errors = deque()
        t = time.time()
        for i in range(num_batch_loop):
            _, c, pred_batch = self.sess.run([optimizer, cost, pre],
                                             feed_dict={self.userBatch: self.trainData[
                                                                        i * self.batchSize:(i + 1) * self.batchSize, 0],
                                                        self.itemBatch: self.trainData[
                                                                        i * self.batchSize:(i + 1) * self.batchSize, 1],
                                                        self.ratingBatch: self.trainData[
                                                                          i * self.batchSize:(i + 1) * self.batchSize,
                                                                          2:]})
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)

            # print type(pred_batch)
            errors.append(np.mean(
                np.power(pred_batch - self.trainData[i * self.batchSize:(i + 1) * self.batchSize, 2], 2)))
        if self.trRow % self.batchSize:
            _, c, pred_batch = self.sess.run([optimizer, cost, pre],
                                             feed_dict={self.userBatch: self.trainData[(i + 1) * self.batchSize:, 0],
                                                        self.itemBatch: self.trainData[(i + 1) * self.batchSize:, 1],
                                                        self.ratingBatch: self.trainData[(i + 1) * self.batchSize:,
                                                                          2:]})
            pred_batch = np.clip(pred_batch, self.minRating, self.maxRating).reshape(-1)
            # print type(pred_batch)
            errors.append(np.mean(
                np.power(pred_batch - self.trainData[(i + 1) * self.batchSize:, 2], 2)))

        trRMSE = np.sqrt(np.mean(errors))
        return trRMSE, time.time() - t

    def train(self, mod = 'CPU',num_CPU = 1,num_GPU = 0,epochs = None,shuffle = False):
        if epochs == None:
            epochs = self.epochs
        self.RMSEtr = []
        self.RMSEts = []
        # bestRmse  = 100
        self.wHis = []
        # self.sess.run(merged)
        pre, cost, optimizer = self.creatSess(mod=mod, num_CPU=num_CPU,num_GPU=num_GPU)
        self.logger.info('users = %d, items = %d, K = %d, u_reg = %.1e, i_reg = %.1e, learningRate = %.1e, batchSize = %d, epochs = %d,  minRating = %d, maxRating = %d,optimizer = %s'
              %(self.users, self.items, self.K, self.uReg, self.iReg, self.learningRate, self.batchSize, self.epochs, self.minRating, self.maxRating, optimizer.name))
        tsMAE,tsRMSE, testTime, prediction = self.testIter(pre)
        # RMSEts.append(tsRMSE)
        bestRmse = tsRMSE
        self.testPrediction = np.concatenate(prediction)
        self.logger.info("Init Test loss:" + str(round(tsRMSE, 4)) + ' Test time: ' + str(round(testTime, 3)))

        for epoch in range(epochs):
            # writer.add_summary(summary,epoch)
            trRMSE, trainTime = self.trainIter(pre, cost, optimizer,shuffle)
            self.RMSEtr.append(trRMSE)
            tsMAE, tsRMSE, testTime, prediction = self.testIter(pre)
            self.RMSEts.append(tsRMSE)
            precisions, recalls, ndcgs = self.rankTest(pre)
            if tsRMSE < bestRmse:
                bestRmse = tsRMSE
                self.testPrediction = np.concatenate(prediction)
                self.save(bestRmse,tsMAE)
            self.logger.info("Epoch " + str(epoch + 1) + " completed out of " + str(self.epochs)
                             + "; Train loss:" + str(round(trRMSE, 4))
                             + ' Train time: ' + str(round(trainTime, 3))
                             + "; Test loss:" + str(round(tsRMSE, 4))
                             + ", Test MAE:" + str(round(tsMAE, 4))
                             + ', precsion: %.4f, recall: %.4f, NDCG: %.4f ' % (
                             np.mean(precisions), np.mean(recalls), np.mean(ndcgs))
                             + ' Test time: ' + str(round(testTime, 3)))

        err = self.testPrediction - self.testData[:, 2]
        self.logger.info('best RMSE : %s' % str(round(np.sqrt(np.power(err, 2).mean()), 5)))
        self.logger.info('best MAE : %s' % str(round(np.abs(err).mean(), 5)))

