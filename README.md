# Testing embedOR algorithm on 15 datasets that have been previously visualized using TSNE and/or UMAP

Notes: 
* The only non-scRNA-seq datasets are Wikipedia (text) and CIFAR (images), both of which were embedded into vector representations using machine learning models.
* ScRNA-seq datasets were titled according to the primary author’s last name. Preprocessing scripts were written for most datasets, though some were more preprocessed than others. When possible, datasets were preprocessed according to the methods detailed in the reference paper. Otherwise, datasets underwent the standard Seurat workflow consisting of filtering, normalizing counts, log-transforming, identifying HVGs, and PCA. 
* Raw data and preprocessing scripts for each dataset are available in their respective folders (with the exception of Tasic and Kanton, which were entirely preprocessed). All data used to generate plots are stored in the preprocessed_data folder.
* All plots were generated with n_points = 5000, set.seed = 42.
