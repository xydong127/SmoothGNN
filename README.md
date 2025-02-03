
## Tested Environment

- Ubuntu 22.04
- [Python 3.8.0](https://www.anaconda.com/products/individual#Downloads)
- [Sklearn 1.3.2](https://scikit-learn.org/stable/install.html)
- [Pytorch 2.1.0](https://pytorch.org/get-started/locally/#linux-installation)
- [Numpy 1.24.4](https://numpy.org/install/)
- [Torch_geometric 2.5.2](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [Scipy 1.10.1](https://scipy.org/)
- [Dgl 2.1.0](https://www.dgl.ai/pages/start.html)

## Datasets

Download zip files from [GADbench](https://github.com/squareRoot3/GADBench) (or import from dgl) and unzip them in datasets/. 

**Directory Structure**

```
├── datasets
│   ├── reddit
│   │   ├── reddit (different datasets may have different types)
│   ├── main.py  
│   ├── utils.py
│   ├── name.py
```
Use main.py to generate index. 

**Example**
```
python main.py --data reddit
```

## Experiments

Conduct experiments in code/

**Parameters**
As described in paper. 

**Example**
```
python main.py --data reddit
```
