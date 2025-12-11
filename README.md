# MSMP++ – Product Matching & Similarity Pipeline
MSMP++ is a Python-based framework for product matching, clustering, and attribute analysis.
It processes product datasets, extracts token features, identifies stop words, builds vocabularies, and performs similarity scoring using clustering and Bayesian optimization.

## Features
- Token extraction from product titles and attributes
- Stop-word identification using frequency thresholds
- Vocabulary indexing for text-based product features
- Clustering via Agglomerative Clustering
- Parameter tuning using Bayesian Optimization
- Utility tools for:
    - string cleaning
    - dictionary handling
    - similarity scoring
    - plotting and visualization

## Project Structure
- MSMP++.py          # Main Python module containing all functionality
- README.md          # Project documentation

## Requirements
Install required dependencies:
pip install numpy matplotlib scikit-learn scipy bayesian-optimization

## Dataset

The product dataset used for testing and development can be downloaded from:

https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip

## Technologies Used
- Python 3.10+
- NumPy: numerical computations
- Matplotlib: data visualization
- SciPy: interpolation and optimization
- Scikit-Learn: clustering
- Bayesian Optimization (bayes_opt): hyperparameter optimization

## License
This project is licensed under the MIT License 
