# DANSER-WWW-19

This repository holds the codes for [Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems. (WWW 2019)](https://arxiv.org/abs/1903.10433). 
The requirements are Python 3.6 and Tensorflow 1.7.0.

We also release the public dataset Epinions used in our work and the trained model.

To reproduce the results in our paper, you can run

    python test.py

To train the model, you can run

    python train.py
    
and you will get similar results after about 20 epoches. 

The data_preparation.py is to preprocess the dataset and outputs two .pkl files in /data. The input.py generates mini-batch samples.
The eval.py contains the calculation for evaluation metrics. The model.py implements the network model proposed in our paper.

If you use this code as part of any published research, please cite the following paper:

```
@inproceedings{DANSER-WWW-19,
  author    = {Qitian Wu and Hengrui Zhang and Xiaofeng Gao and Peng He and
               Paul Weng and Han Gao and Guihai Chen},
  title     = {Dual Graph Attention Networks for Deep Latent Representation of Multifaceted
               Social Effects in Recommender Systems},
  booktitle = {The World Wide Web Conference, {WWW} 2019, San Francisco, CA, USA,
               May 13-17, 2019},
  year      = {2019}
  }
```

For more details, you can refer to our paper.

The Epinions data is provided by the work

"eTrust: Understanding trust evolution in an online world" in KDD'2012.

