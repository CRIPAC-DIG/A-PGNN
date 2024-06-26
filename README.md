# A-PGNN

<img src="a_pgnn.png" alt="model" style="zoom: 50%;" />

This is the code for the TKDE Paper: [Personalized Graph Neural Networks with Attention Mechanism for Session-Aware Recommendation](https://ieeexplore.ieee.org/abstract/document/9226110).

## Usage

### Preparation data and code
Here are two datasets we used in our paper.

* Xing http://2016.recsyschallenge.com/
* Reddit https://www.kaggle.com/colemaclean/subreddit-interactions

The processed data can be downloaded: https://www.dropbox.com/sh/hwx2347ir1worag/AABJK6IBXHNBlbvrvKqw94YKa?dl=0

### Generate data

You need to run the file ```record.py``` first to preprocess the data to generate the tf.record formart data for training and test.

For example:
```python record.py --dataset=all_data --data=xing --adj=adj_all --max_session=50```. This will creat ```xing/``` in ```datasets/```.

This code can be given the following command-line arguments:

```--dataset:``` choose to use fully data or samples, if ```all_data```: use fully data, if ```None``` or ```sample```: use sample data.

```--data:```  the name of data set, we can choose ```xing``` or ```reddit```.

```--graph:```  graph neural network, default set is ```ggnn```.

```--max_session:```  the maximum length of historical sessions.

```--max_length:```  the maximum length of current session.

```--last:```  if ```True```, the last one for testing, else, the next 20% for testing.

### Training and Testing 

Then you can run the file ```train_last.py``` to train the model and test.

For example: ```python train_last.py --data=xing --mode=transformer --user_ --adj=adj_all --dataset=all_data --hiddenSize=100```

This code can be given the following command-line arguments:

```--dataset:``` choose to use fully data or samples, if ```all_data```: use fully data, if ```None``` or ```sample```: use sample data.

```--data:```  the name of data set, we can choose ```xing``` or ```reddit```.

```--user_:```  whether to use user embedding. 

```--max_session:``` the maximum length of historical sessions.

```--max_length:``` the maximum length of current session.

```--adj:```  if ```adj_all```, use normalized weights for adjacency matrices, else, use binary adjacency matrix. 

```--batchSize:``` batchsize.

```--epoch:```  epoch.

```--lr:```  learning rate.

```--buffer_size:```  the maximum number of elements that will be added to the buffer. For details, see the use of ```tf.record```, for Xing, we set 200000, Reddit is 100000.


## Requirements

- Python 3.6.5
- Tensorflow-gpu 1.10.0 

## Citation

Please cite our paper if you use the code:

```
@article{zhang2020personalized,
  title={Personalized Graph Neural Networks with Attention Mechanism for Session-Aware Recommendation},
  author={Zhang, Mengqi and Wu, Shu and Gao, Meng and Jiang, Xin and Xu, Ke and Wang, Liang},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2020},
  publisher={IEEE}
}
```
