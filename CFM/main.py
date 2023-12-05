from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from BMF import BMF
from baseMF import baseMF
from CFM import CFM
from utils import getRMSE,getMAE
import pandas as pd
import time
#import os
import sys
import logging



def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    #parser.add_argument('--input', #required=True,
    #                    help='Input graph file')
    parser.add_argument('-d','--dataset',default = 'ta',
                        help='The dataset name')
    parser.add_argument('--K', default=50, type=int,
                        help='Number of latent vectors')
    parser.add_argument('--criteriaNum',  default=6, type=int,
                        help='Number of criteria ratings')
    parser.add_argument('--cv', default=10, type=int,
                        help='The cv in datasets')
    parser.add_argument('--percent', default='02',
                        help='The percent in trainning.')
    parser.add_argument('-e','--epochs', default=5, type=int,
                        help='The training epochs.')
    parser.add_argument('--batchSize', default=1, type=int,
                        help='The batch size of training.')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='The learning rate.')
    parser.add_argument('--learner', default = 'adam',
                        help='the optimazation algorithms')
    parser.add_argument('--maxR', default=5.0, type=float,
                        help='The maximum rating of datasets')
    parser.add_argument('--minR', default=1.0, type=float,
                        help='The minimum rating of datasets')
    parser.add_argument('--method', required=True, choices=[
        'BMF',
        'CFM',
    ], help='The learning method')
    parser.add_argument('--biasR', default=0.01, type=float,
                        help='The regularization of biasd in BMF et. al.')
    parser.add_argument('--uR', default=0.01, type=float,
                        help='The regularization of users\' latent vector.')
    parser.add_argument('--iR', default=0.01, type=float,
                        help='The regularization of items\' latent vector.')
    parser.add_argument('--regressionR', default=0.01, type=float,
                        help='The regularization of regression weights.')
    parser.add_argument('--criR', default=0.01, type=float,
                        help='The regularization of criteria weights for MSVD.')
    parser.add_argument('--reg', default=None, type=float,
                        help='The regularization for SVDPP.')
    parser.add_argument('--lam', default=0.01, type=float,
                        help='The effect of criteria rating in CFM et. al.')
    parser.add_argument('--share', default='ind', choices=['user','item','ind'],
                        help='sharing users\' of items\' in CFM model.')
    parser.add_argument('--coreShape', default='[50,50,3]', type=eval,
                        help='The shape of core for MSVD model.')
    parser.add_argument('--saveThreshold', default=0.89, type=float,
                        help='The Threshold for saving model.')
    parser.add_argument('--CPU', default=1, type=int,
                        help='The numbers of GPU cores.')
    parser.add_argument('--GPU', default=0, type=int,
                        help='The numbers of GPU cores.')
    args = parser.parse_args()


    return args


def main(args):
    t1 = time.time()
    train = pd.read_csv('datasets/%s.cv%d.%s.base'%(args.dataset,args.cv,args.percent), sep='\t', header=None)
    test = pd.read_csv('datasets/criteriaSplit2/%s.cv%d.%s.test.col0'%(args.dataset,args.cv,args.percent), sep='\t', header=None)
    items_num = train[1].max()+1
    user_num = train[0].max()+1
    sStr = '%s_K%d_cv%d_percent%s'%(args.dataset,args.K,args.cv,args.percent)
    logroot = 'logs/'
    logName = r'%s%s_%s.log'%(logroot,sStr,time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))#logroot +sStr + '_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  +".log"
    print('log save as: %s'%(logName))
    
    if args.method == 'BMF':
        A = BMF(K=args.K,items = items_num,users = user_num, epochs=args.epochs, iReg=args.iR, uReg = args.uR, biasReg =args.biasR, reg = args.reg,
         learningRate=args.lr, batchSize=args.batchSize, saveStr = sStr, log = logName, optimizer=args.learner, th = args.saveThreshold)
    
    
    elif args.method == 'CFM':
        if args.share == 'ind':
            A = CFM(K=args.K,items = items_num,users = user_num,epochs=args.epochs,regressionReg=args.regressionR, iReg=args.iR, criteriaNum = args.criteriaNum, reg = args.reg,
                uReg = args.uR, biasReg =args.biasR, learningRate=args.lr, lambd = args.lam, batchSize=args.batchSize,saveStr = sStr,log = logName, optimizer=args.learner, th = args.saveThreshold)

        elif args.share == 'user':
            A = SCMF(K=args.K,items = items_num,users = user_num,epochs=args.epochs,regressionReg=args.regressionR, iReg=args.iR, criteriaNum = args.criteriaNum,shared = 'u',
                reg = args.reg, uReg = args.uR, biasReg =args.biasR, learningRate=args.lr, lambd = args.lam, batchSize=args.batchSize,saveStr = sStr,log = logName, optimizer=args.learner, th = args.saveThreshold)

        elif args.share == 'item':
            A = SCMF(K=args.K,items = items_num,users = user_num,epochs=args.epochs,regressionReg=args.regressionR, iReg=args.iR, criteriaNum = args.criteriaNum,shared = 'i',
                reg = args.reg, uReg = args.uR, biasReg =args.biasR, learningRate=args.lr, lambd = args.lam, batchSize=args.batchSize,saveStr = sStr,log = logName, optimizer=args.learner, th = args.saveThreshold)

    A.getData(train = train,test = test)
    #print('args.GPU',args.GPU)
    A.train(mod='cpu',num_CPU=args.CPU,num_GPU = args.GPU)
    print('RMSE:')
    bestRMSE = getRMSE(A.testPrediction,test[2])
    print(bestRMSE)
    print('MAE:')
    MAE = getMAE(A.testPrediction,test[2])
    print(MAE)
    pd.DataFrame(A.testPrediction).to_csv('result/%s_%s_K%d_cv%d_RMSE%s_MAE%s.csv' % (A.__class__.__name__,args.dataset,args.K, args.cv,round(bestRMSE, 4),round(MAE, 4)),
                                          header=None, index=None)
    print('---------------------------Finish--------------------------------------')


if __name__ =='__main__':
    main(parse_args())
