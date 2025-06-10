# DPPC/Fluorinated Compound Monolayer Miscibility Analysis

This repository contains scripts and example data for the feature extraction, multivariate analysis, and Bayesian logistic regression modeling described in:

> Nakahara H, Shibata O. "A Review of the Miscibility of Binary Monolayers Composed of DPPC and Fluorinated Compounds: Insights from Multivariate and Bayesian Modeling" (2024)

## Contents

- `src/main_analysis.py`: Main script for feature engineering, PCA, correlation analysis, and Bayesian logistic regression (PyMC).
- `example_data/example_data.csv`: Example input data with the same structure as Table 1 in the review.
- `requirements.txt`: List of required Python packages.

## How to use

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the analysis script:

    ```bash
    python src/main_analysis.py
    ```

3. When prompted, select your CSV file (or use the example in `example_data/`).  
   The script will generate features, perform PCA/correlation analysis, and fit a Bayesian logistic regression model.

## Notes

- Input data must include isomeric SMILES for both the fluorinated compound (F) and DPPC, plus experimental conditions and miscibility labels.
- The code calculates molecular descriptors using RDKit, generates delta/ratio features, and fits a Bayesian model using PyMC.
- Example figures and statistical summaries will be output to the console and as images.

## Citation

If you use this code, please cite:

> Nakahara H, Shibata O. "A Review of the Miscibility of Binary Monolayers Composed of DPPC and Fluorinated Compounds: Insights from Multivariate and Bayesian Modeling" (2024)

## License

MIT License
