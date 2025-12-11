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

## Related Work 
This code improves and builds on the ideas of the following related papers:
- van Bezu, R., Borst, S., Rijkse, R., Verhagen, J., Frasincar, F., Vandic, D.: Multi-
component similarity method for web product duplicate detection. In: 30th Sympo-
sium on Applied Computing (SAC 2015). pp. 761–768. ACM (2015
- van Dam, I., van Ginkel, G., Kuipers, W., Nijenhuis, N., Vandic, D., Frasincar, F.:
Duplicate detection in web shops using LSH to reduce the number of computations.
In: 31th ACM Symposium on of Applied Computing (SAC 2016). pp. 772–779. ACM
(2016)
- Hartveld, A., Keulen, M., Mathol, D., Noort, T., Plaatsman, T., Frasincar, F.,
Schouten, K.: An LSH-Based Model-Words-Driven Product Duplicate Detection
Method, pp. 409–423 (05 2018)


