# Experiment 5 — Feature Selection Techniques

This repository contains the Python implementation and Jupyter Notebook for Experiment 5, using the same Indian subdivision-wise rainfall CSV dataset as the previous experiments.

## Aim
Identify relevant features using statistical and machine-learning-based feature selection techniques, compare the selected subsets, and evaluate a model using all features versus selected features.

## Dataset and target
- Dataset file: `Sub_Division_IMD_2017(2).csv`
- Target variable: `ANNUAL` rainfall
- Predictors: the 12 monthly rainfall columns (`JAN` through `DEC`)
- `SUBDIVISION` and `YEAR` are not used as numeric predictors.
- Seasonal aggregate columns (`JF`, `MAM`, `JJAS`, `OND`) are excluded to avoid target leakage/redundant information because they overlap with annual rainfall.

**Data suitability note:** This source CSV contains environmental rainfall measurements rather than direct autonomous-system sensor data. It is used here to demonstrate the feature-selection workflow on the same dataset.

## Techniques implemented
1. Data loading, missing-value inspection and median imputation.
2. Pearson correlation analysis and correlation-based feature selection.
3. Mutual information filter method.
4. Recursive Feature Elimination (RFE) wrapper method.
5. Random Forest feature importance (embedded method).
6. Comparison of selected features across methods.
7. Random Forest regression comparison using all 12 monthly predictors versus the top 6 tree-ranked predictors.
8. Evaluation using MAE, RMSE, R² and model fit time.

## Files
- `Experiment_5_Feature_Selection.ipynb` — notebook with explanations, code, outputs and plots.
- `Experiment_5_Feature_Selection.py` — raw Python implementation.
- `Sub_Division_IMD_2017(2).csv` — dataset.

## Requirements
Python 3.x, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter.

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run
Keep the CSV in the same directory as the notebook/script.

```bash
python Experiment_5_Feature_Selection.py
```

Or open the notebook in Jupyter Notebook / Google Colab and run all cells.

## Notes
- The train/test split uses `random_state=42` for reproducibility.
- Feature selection and model evaluation are performed using training/test partitions.
- The same Random Forest model configuration is used for the all-feature and selected-feature comparison.
- Results depend on the supplied dataset and train/test split.
