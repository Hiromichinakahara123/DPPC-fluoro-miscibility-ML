# DPPC/Fluorinated Compound Monolayer Miscibility Analysis

This repository contains scripts for feature extraction, multivariate analysis, and Bayesian logistic regression modeling, as described in:

> Nakahara H, Shibata O. "A Review of the Miscibility of Binary Monolayers Composed of DPPC and Fluorinated Compounds: Insights from Multivariate and Bayesian Modeling" (2024)

## Contents

- `src/main_analysis.py`: Main script for feature engineering, PCA, correlation analysis, and Bayesian logistic regression (PyMC).
- `requirements.txt`: List of required Python packages.

## How to use

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Prepare your data as a CSV file, following the same format as described in Table 1 of the review.  
   Required columns:
   - isomeric SMILES(F)
   - isomeric SMILES(DPPC)
   - ionic strength
   - pH
   - temperature(K)
   - (Thermodynamical) Miscibility

3. Run the analysis script, specifying your data file:

    ```bash
    python src/main_analysis.py your_data.csv
    ```

   The script will generate features, perform PCA/correlation analysis, and fit a Bayesian logistic regression model.

## Notes

- Input data **is not included** due to data policy.  
  Please prepare your own CSV according to the required format.
- The code calculates molecular descriptors using RDKit, generates delta/ratio features, and fits a Bayesian model using PyMC.
- Example figures and statistical summaries will be output to the console and as images.

## Citation

If you use this code, please cite:

> Nakahara H, Shibata O. "A Review of the Miscibility of Binary Monolayers Composed of DPPC and Fluorinated Compounds: Insights from Multivariate and Bayesian Modeling" (2024)

## License

MIT License
