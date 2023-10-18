## Dataset

The datasets and pretrain embeddings can be downloaded from [GANA](https://github.com/ngl567/GANA-FewShotKGC).

## How to Run

### NELL-One

```python
python main.py --negsize 16 --few 1
```

We uploaded the FIIN model parameters trained on the NELL-One dataset.  Please run as follows:

```python
python main.py --test
```

### WIKI-One

```python
python main.py --datapath dataset/WIKI/ --lr 3e-4 --prefix wiki
```



