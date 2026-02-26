# EmbedOR
## Official Implementation of the paper:  [*"EmbedOR: Provable Cluster-Preserving Visualizations with Curvature-Based Stochastic Neighbor Embeddings"*](https://arxiv.org/pdf/2509.03703)

## Getting Started

### Installation

Clone the repository:

``` bash
git clone https://github.com/kathyzxu/embedOR.git
cd embedOR
```

Create and activate a virtual environment:

``` bash
python3 -m venv env
source env/bin/activate
export PYTHONPATH=/path/to/embedor
```

Upgrade pip and install dependencies:

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Test
You can test on the sample dataset:
``` bash
python scripts/run.py preprocessed_data/chimp.npy --labels preprocessed_data/chimp.labels.npy --n_points 5000 --seed 42 --layout torch
```

Or on any .npy  file with the following command: 
``` bash
python scripts/run.py <np_file> [--labels <labels_file>] [--n_points N] [--seed SEED] [--layout {numpy,torch}]
```
