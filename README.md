
This is an implement of Collective Factor Model (CFM) for Multi-Criteria Recommender Systems. 
[Predicting ratings in multi-criteria recommender systems via a
collective factor model](https://demalworkshop.github.io/www2021/papers/predictingratings.pdf) and [Improving Rating Prediction in Multi-Criteria Recommender Systems Via a Collective Factor Model](https://www.researchgate.net/publication/370264368_Improving_Rating_Prediction_in_Multi-criteria_Recommender_Systems_via_a_Collective_Factor_Model)


# Quickstart
- Clone this repo.
- enter the directory where you clone it, and run the following code
    ```bash
    pip install -r requirements.txt
    python -m CFM --method BMF
    ```
## Options
You can check out the other options available to use with *CFM* using:

    python -m CFM --help

- -d, --dataset, The dataset name; the default is ta;
- --K, Number of latent vectors; the default is 50;
- --criteriaNum, Number of criteria ratings; the default is 6;
- --cv, ,The cv in datasets; the default is 10;
- --percent, The percent in trainning; the default is 02;
- -e, --epochs, The training epochs; the default is 1;
- --batchSize, The batch size of training; the default is 1;
- --lr, The learning rate; the default is 0.1;
- --learner, the optimazation algorithms; the default is adam;
- --maxR, The maximum rating of datasets; the default is 5.0;
- --minR, The minimum rating of datasets; the default is 1.0;
- --method, The learning method (BMF, CFM); 
- --biasR, The regularization of biasd in BMF et. al; the default is 0.01;
- --uR, The regularization of users\ latent vector; the default is 0.01;
- --iR, The regularization of items\ latent vector; the default is 0.01;
- --regressionR, The regularization of regression weights; the default is 0.01;
- --reg, set regularization for all weights ; the default is 0.01;
- --lam, The effect of criteria rating in CFM et. al; the default is 0.01;
- --share, sharing users' or items' for CFM model (user ,item ,ind); the default is ind;
- --saveThreshold, The Threshold for saving model,the default is 0.89;
- --CPU, The numbers of CPU cores, the default is 1;
- --GPU, The numbers of GPU cores, the default is 0;


## Citing
If you find *CFM* is useful for your research, please consider citing the following papers:

        @inproceedings{fan2021predicting,
        title={Predicting ratings in multi-criteria recommender systems via a collective factor model},
        author={Fan, Ge and Zhang, Chaoyun and Chen, Junyang and Wu, Kaishun},
        booktitle={DeMal@ The Web Conference},
        pages={1--6},
        year={2021}
        }
        
        @ARTICLE{fan2023improving,
          author={Fan, Ge and Zhang, Chaoyun and Chen, Junyang and Li, Paul and Li, Yingjie and Leung, Victor C. M.},
          journal={IEEE Transactions on Network Science and Engineering}, 
          title={Improving Rating Prediction in Multi-Criteria Recommender Systems Via a Collective Factor Model}, 
          year={2023},
          volume={},
          number={},
          pages={1-11},
          doi={10.1109/TNSE.2023.3270910}}
