# Matrix Decomposition Methods for Face Recognition
Project completed for the final assessment in the Fall 2018 semester of APMTH 205: Advanced Scientific Computing--Numerical Methods
Eric Sun and Eric Wasserman

To access 1D-PCA, SVD, Sparse PCA, and NMF models please refer to the functions in the main Jupyter notebook in the top directory.
To access 2D-PCA models please refer to the Python scripts in the "Code" subdirectory.

Our face recognition pipeline consists of:
1. Pre-processing image data (alignment, cropping, masking)
2. Decomposiing training and testing image sets into matrix components (e.g. SVD, PCA, 2D-PCA, Sparse PCA, NMF)
3. Classifying projected testing data (e.g. nearest neighbor, nearest class mean, k-means clustering assignment)
