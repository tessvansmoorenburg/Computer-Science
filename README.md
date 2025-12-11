# MSMP++ – Scalable Product Duplicate Detection

MSMP++ is a Python-based framework for scalable product duplicate detection (Entity Resolution). It implements an enhanced version of the Multi-component Similarity Method (MSM) by integrating Locality Sensitive Hashing (LSH) with frequency-based token pruning to improve scalability and precision.

## Features
- **Frequency-Based Pruning:** Automatically identifies and removes high-frequency, low-discriminative tokens (stop words) to prevent LSH bucket overflow[cite: 43].
- **LSH & MinHash:** Efficient candidate generation using MinHash signatures and LSH banding
- **Token Extraction:** Advanced regex-based extraction for model words and attributes.
- **Clustering:** Hierarchical Agglomerative Clustering for final pair prediction.
- **Optimization:** Hyperparameter tuning using Bayesian Optimization.
- **Visualization:** Automatic plotting of Pair Quality, Completeness, and F1 curves.

## Project Structure
- `MSMP++.py`       # Main Python script containing the pipeline, optimization, and evaluation loop.
- `README.md`        # Project documentation.

## Requirements
Ensure you have Python 3.10+ installed. Install the required dependencies:
`pip install numpy matplotlib scikit-learn scipy bayesian-optimization` 

## Dataset

The project is designed to work with the TV product dataset.

1. Download the dataset from: https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip
2. Unzip the file to get TVs-all-merged.json.

## How to Run
Configure Data Path: Open MSMP++.py and locate the configuration section at the bottom (around line 522). Update the JSON_PATH variable to point to your local TVs-all-merged.json file:
`#>>>>> CONFIGURATION: SET YOUR DATA PATH HERE <<<<<
JSON_PATH = "path/to/your/TVs-all-merged.json"` 

Execute the Script: Run the full pipeline (Pruning -> Optimization -> Evaluation -> Plotting):
python MSMP++.py
Note: The script performs Bayesian Optimization and 5-fold bootstrap validation. A full run may take several minutes. If you want to run it without Bayesian Optimization you can run it with the following optimized parameters: `EPSILON = 0.452, MU = 0.8, ALPHA = 0.572, BETA = 0.0, GAMMA = 0.9` 

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


