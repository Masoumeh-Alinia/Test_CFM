
import tensorflow as tf

#from tensorflow.keras.regularizers import l1,l2

def l1(x):
    return tf.keras.regularizers.l1(l=0.1)(x)

def regressionModel(_mod = 'Linear',reg = 0.01,**kwargs):
    if _mod.lower() == 'linear':
        model = tf.layers.Dense(1,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),#stddev=0.01
                                #kernel_initializer=tf.ones_initializer(),
                                kernel_regularizer=tf.keras.regularizers.l2(l=reg),
                                bias_regularizer=tf.keras.regularizers.l2(l=reg),
                                **kwargs)
        #model = tf.reshape(model,[-1])
    elif _mod.lower() == 'mean':
        model = lambda x: tf.reduce_mean(x, 1)
    elif _mod.lower() == 'SVR':
        model = 1

    return model