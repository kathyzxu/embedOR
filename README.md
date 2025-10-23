# Testing embedOR algorithm on 15 datasets that have been previously visualized using TSNE and/or UMAP

Notes: 
* The only non-scRNA-seq datasets are Wikipedia (text) and CIFAR (images), both of which were embedded into vector representations using machine learning models.
* ScRNA-seq datasets were titled according to the primary author’s last name. Preprocessing scripts were written for most datasets, though some were more preprocessed than others. When possible, datasets were preprocessed according to the methods detailed in the reference paper. Otherwise, datasets underwent the standard Seurat workflow consisting of filtering, normalizing counts, log-transforming, identifying HVGs, and PCA. 
* Preprocessing scripts for each dataset are available with the exception of Tasic and Kanton, which were entirely preprocessed prior. Consult datasets_summary.csv for information about each dataset.
* All plots were generated with n_points = 5000, set.seed = 42.
* Permutation tests measure low-energy edge expansion (Tables 3, 6) and high-energy edge contraction (Tables 7,8). All  tests were run with 10,000 permutations and 5000 subsampled points for datasets with more than 5000 samples. 
