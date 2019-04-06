# DANSER-WWW-19
This repository hosts the experimental code for WWW 2019 full paper "Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems".

We use Python 3.6 with Anaconda3 and Tensorflow-gpu 1.7.0 to implement the codes. 
The code is for Epinions dataset, which is provided by paper:
Trust-aware recommender systems. P Massa, P Avesani. Proceedings of the 2007 ACM conference on Recommender systems, 17-24

You can first run the data_preparation.py file for dataset split and feature preparation.
Then you can run the train.py file, a top manuscript that runs the process of training and testing.
The model.py file implements the network model of DANSER, the input.py file is for feeding mini-batch data to the network, and the eval.py contains the computation for several evaluation metrics.

More detailed instructions will be provided soon.

For any questions, you can report issue here.

