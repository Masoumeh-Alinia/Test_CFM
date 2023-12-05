import tensorflow as tf
import numpy as np
#import pandas as pd
from baseMF import baseMF
import time
#import matplotlib.pyplot as plt
# from remtime import *
from collections import deque


# from remtime import printTime
class BMF(baseMF):
   
    def __init__(self, biasReg=0.01, **kwargs):
        super(BMF, self).__init__(**kwargs)
        self.biasReg = biasReg
        self.userBias = tf.Variable(tf.zeros([self.users]))
        self.itemBias = tf.Variable(tf.zeros([self.items]))

    def buildGraph(self):
        # userBatch, itemBatch, ratingBatch = it.get_next()
        self.globalBias = tf.constant(self.trainData[:, 2:3].mean(axis=0), dtype=tf.float32)
        userWeightBatch = tf.nn.embedding_lookup(self.userWeight, self.userBatch)
        itemWeightBatch = tf.nn.embedding_lookup(self.itemWeight, self.itemBatch)

        userBiasBatch = tf.nn.embedding_lookup(self.userBias, self.userBatch)
        itemBiasBatch = tf.nn.embedding_lookup(self.itemBias, self.itemBatch)

        output = tf.reduce_sum(tf.multiply(userWeightBatch, itemWeightBatch), 1)

        output = tf.add(output, self.globalBias)
        output = tf.add(output, userBiasBatch)
        prediction = tf.add(output, itemBiasBatch)

        base = tf.nn.l2_loss(tf.subtract(prediction, self.ratingBatch))

        reg = tf.add(self.uReg * tf.nn.l2_loss(userWeightBatch), self.iReg * tf.nn.l2_loss(itemWeightBatch))

        bReg = tf.add(tf.nn.l2_loss(self.biasReg * userBiasBatch), tf.nn.l2_loss(self.biasReg * itemBiasBatch))
        # bReg = tf.add(bReg, tf.nn.l2_loss(self.globalBias))
        reg = tf.add(reg, bReg)
        cost = tf.add(base, reg)
        # r = tf.clip_by_value(prediction,self.minRating,self.maxRating)
        # error = tf.reduce_mean(tf.pow(tf.subtract(r, self.ratingBatch),2))
        # print ('OK')
        return prediction, cost  # ,error



