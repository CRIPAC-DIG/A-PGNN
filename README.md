# A-PGNN
The code and dataset for our paper: Personalizing Graph Neural Networks with Attention Mechanism for Session-based Recommendation (https://arxiv.org/abs/1910.08887). We have implemented our methods in Tensorflow.

Here are two datasets we used in our paper.

* Xing 
* Reddit

## Usage 

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

For example: ```python train_last.py --data=xing --mode=transformer --user_ --adj=all_adj --dataset=all_data --hiddenSize=100```

This code can be given the following command-line arguments:

```--dataset:``` choose to use fully data or samples, if ```all_data```: use fully data, if ```None``` or ```sample```: use sample data.

```--data:```  the name of data set, we can choose ```xing``` or ```reddit```.

```--user_:```  whether to user user embedding. 

```--max_session:``` the maximum length of historical sessions.

```--max_length:``` the maximum length of current session.

```--adj:```  if ```adj_all```, use normalized weights for adjacency matrices, else, use binary adjacency matrix. 

```--batchSize:``` batchsize

```--epoch:```  epoch

```--lr:```  learning rate

```--buffer_size:``` 






## Requirement
* Python3.6.5
* Tensorflow-gpu1.10.0 

## Citation

```
  @article{wu2019personalizing,
  title={Personalizing Graph Neural Networks with Attention Mechanism for Session-based Recommendation},
  author={Wu, Shu and Zhang, Mengqi and Jiang, Xin and Ke, Xu and Wang, Liang},
  journal={arXiv preprint arXiv:1910.08887},
  year={2019}
}
```




