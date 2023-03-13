import tensorflow as tf
import numpy as np
def regressionModel(_mod = 'Linear'):
    if _mod == 'Linear':
        model = tf.layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    elif _mod == 'Mean':
        model = lambda x: tf.reduce_mean(x, 1)
    return model

def evaluate(pred,true,metric = ['RMSE','MAE']):

    if 'RMSE' in metric:
        getRMSE(pred,true)
    if 'MAE' in metric:
        getRMSE(pred,true)
    return 


def getRMSE(pred,true):
    return np.sqrt(np.power(pred - true, 2).mean())

def getMAE(pred,true):
    return np.abs(pred - true).mean()
