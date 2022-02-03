# Backdoor Attacks-Resilient Aggregation based on Robust Filtering of Outliers in Federated Learning for image classification

Federated Learning (FL) is a distributed machine learning paradigm vulnerable to different kind of adversarial attacks, since its distributed nature and the inaccessibility of the data by the central server.  In this work, we focus on model-poisoning backdoor attacks, because they are characterized by their stealth and effectiveness.  We claim that the model updates of the clients of a federated learning setting follow a Gaussian distribution, and those ones withan outlier behavior in that distribution are likely to be adversarial clients. We propose a new federated aggregation operator called Robust Filtering of one-dimensional Outliers (RFOut-1d), which works as a resilient defensive mechanism to model-poisoning backdoor attacks. RFOut-1d is based on an univariate outlier detection method that filters out the model updates of the adversarial clients. The results on three federated image classification dataset show that RFOut-1d dissipates the impact of the backdoor attacks to almost nullifying them throughout all the learning rounds, as well as it keeps the performance of the federated learning model and it outperforms that state-of-the-art defenses against backdoor attacks.

In this repository, we provide the implementation of RFOut-1d in two Federated Learning frameworks, namely: Flower and Sherpa.ai Federated Learning.  Likewise, we show its behavior in each implementation on an image classification problem with the [ EMNIST Digits datset](https://www.nist.gov/itl/products-and-services/emnist-dataset).


## Implementation in Flower

We provide the implementation in  [Flower](https://flower.dev/).

**Requirements**. 

* The Flower framework. Follow the [official instructions](https://flower.dev/docs/installation.html) to install it.
* Python version >= 3.6.

**Usage**. You have to follow the following steps to run the image classification experiment with the Flower implementation and to use the code in [this directory](./flower/).

1- Open a terminal and start a server with the RFOut-1d strategy.

```
python ./flower/server.py
```

2- Run the first client in other terminal.

```
python ./flower/client.py
```

3- In different terminals, add as many clients as you want in the federated configuration (min. 2 according to the framework details).

```
python ./flower/client.py
```

The clients will show the results of the training in each learning of round and the server after each aggregation.

We also provide a [Jupyter notebook](./flower/rfout.ipynb) to show its behavior with 2 clients.

## Implementation in Sherpa.ai Federaeted Learninng

We also provide the implementation in [Sherpa.ai Federated Learning](https://github.com/rbnuria/Sherpa.ai-Federated-Learning-Framework.git).

We provide a [Jupyter notebook](./shfl/rfout.ipynb) in which we set up the entire federated setup of a simple image classification experiment and detail the code of the RFOut-1d aggregation mechanism. The [file rfout.py](./shfl/rfout.py) contains the implementation of RFOut-1d.

**Requirements**. 

* The Sherpa.ai FL framework. Clone [this GitHub repository](https://github.com/rbnuria/Sherpa.ai-Federated-Learning-Framework.git).
* Python version >= 3.6.

**Usage**. Once you have clone the Github repository, move the [Jupyter notebook (rfout.ipynb)](./shfl/rfout.ipynb) and the implementation of the aggregation [python file (rfout.py)](./shfl/rfout.py) to the root directory and run all cells of the notebook.

## Citation
If you use this dataset, please cite:

*Citation not available yet*.


## Contact
Nuria Rodríguez Barroso - rbnuria@ugr.es
