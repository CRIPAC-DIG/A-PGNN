# A-PGNN
The code and dataset for our paper: Personalizing Graph Neural Networks with Attention Mechanism for Session-based Recommendation (https://arxiv.org/abs/1910.08887). We have implemented our methods in Tensorflow.

Here are two datasets we used in our paper.

* Xing 
* Reddit

## Usage 

You need to run the file ```record.py``` first to preprocess the data to generate the tf.record formart data for training and test.

For example:
```python record.py --dataset=all_data --data=xing --adj=adj_all --max_session=50```. This will creat ```xing/``` in ```datasets/```.


```
usage: record.py [--dataset ] [--data data_name] [--graph ggnn] [--max_session the length of historical session]

optional arguments:
--dataset:  use fully data or samples
--data:     data name
--graph:    graph neural network 
--max_session: the length of historical sessions
--max_length:  the length of current session
--last: the type of test
```

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




