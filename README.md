# DPEL
A Distribution Programming Example Library of Popular Deep Learning Frameworks


## PyTorch

```python
python -m torch.distributed.launch --nproc_per_node=GPU_nums pytorch_based_distributed_mnist.py
```

## Tensorflow

```python
python tensorflow_based_distributed_mnist.py
```


## Training Performance via GeForce RTX 2080 Ti(x2)

* Dataset: MNIST
* Period: 1 Epoch
* Batch size per GPU for training: 9000
* No validation

|Frameworks|Single|Distributed|
|----------|------|-----------|
|PyTorch   | 5.50s|   2.72s   |
|Tensorflow| 0.65s|   0.27s   |
