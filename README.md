# EmbedOR

**Official implementation of the paper:**  
*"EmbedOR: Provable Cluster-Preserving Visualizations with Curvature-Based Stochastic Neighbor Embeddings"*  
[Read the paper on arXiv](https://arxiv.org/pdf/2509.03703)

## Repository Structure:

embedor/
├─ benchmarking/     # Generated visualizations and stats against other clustering algorithms
├─ sample_data/      # Example dataset
├─ scripts/          # Example scripts and tutorial
├─ src/              # Python modules for EmbedOR
├─ requirements.txt  # Python dependencies
└─ README.md

## Installation

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

## Run

### To run the main script:

``` bash
python scripts/run.py <np_file> [--labels <labels_file>] [--n_points N] [--seed SEED] [--layout {numpy,torch}]
```

Test on chimp dataset:

``` bash
python scripts/run.py preprocessed_data/chimp.npy --labels preprocessed_data/chimp.labels.npy --n_points 1000 --seed 42 --layout numpy
```
