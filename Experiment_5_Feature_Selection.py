import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Sub_Division_IMD_2017(2).csv")
print("Dataset shape:", df.shape)
display(df.head())
print("\\nMissing values:")
display(df.isnull().sum().to_frame("Missing values"))

# 2. Define independent and dependent variables.
# Predict ANNUAL rainfall using monthly rainfall columns only.
# Seasonal aggregates are excluded to avoid target leakage/redundant sums.
feature_cols = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
target_col = "ANNUAL"

data = df[feature_cols + [target_col]].copy()
data = data.dropna(subset=[target_col])
X = data[feature_cols]
y = data[target_col]

print("Independent variables:", feature_cols)
print("Dependent variable:", target_col)
print("Rows available for modelling:", len(data))

# 3. Basic cleaning: median imputation for missing monthly values
imputer = SimpleImputer(strategy="median")
X_clean = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)
print("Missing values in predictors after imputation:", int(X_clean.isna().sum().sum()))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

# 4. Correlation analysis (filter method)
corr = pd.concat([X_clean, y.rename(target_col)], axis=1).corr()
target_corr = corr[target_col].drop(target_col).sort_values(key=np.abs, ascending=False)
print("\\nCorrelation with annual rainfall:")
display(target_corr.to_frame("Correlation with ANNUAL"))
plt.figure(figsize=(10, 7))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Select top 6 features by absolute Pearson correlation (filter selection)
correlation_selected = target_corr.abs().head(6).index.tolist()
print("Top 6 correlation-selected features:", correlation_selected)

# Mutual information filter method, fitted on training data only
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
print("\\nMutual information scores:")
display(mi_series.to_frame("Mutual information"))
mi_selected = mi_series.head(6).index.tolist()
print("Top 6 mutual-information features:", mi_selected)

# 5. Wrapper method: Recursive Feature Elimination (RFE)
rfe_estimator = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=6, step=1)
rfe.fit(X_train, y_train)
rfe_selected = X_train.columns[rfe.support_].tolist()
print("\\nRFE selected features:", rfe_selected)

# 6. Embedded method: tree-based feature importance
tree_model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
tree_model.fit(X_train, y_train)
importance = pd.Series(tree_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\\nRandom Forest feature importance:")
display(importance.to_frame("Importance"))
tree_selected = importance.head(6).index.tolist()
print("Top 6 tree-importance features:", tree_selected)

plt.figure(figsize=(9, 5))
importance.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 7. Compare selected features across methods
selection_comparison = pd.DataFrame({
    "Correlation": pd.Series(correlation_selected),
    "Mutual Information": pd.Series(mi_selected),
    "RFE": pd.Series(rfe_selected),
    "Tree Importance": pd.Series(tree_selected)
})
print("\\nFeature selection comparison:")
display(selection_comparison)

# 8. Train same model with all features and selected features.
# Use the top 6 tree-importance features for the selected-feature model.
def evaluate_model(model, Xtr, Xte, ytr, yte):
    start = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_time = time.perf_counter() - start
    pred = model.predict(Xte)
    return {
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "R2": r2_score(yte, pred),
        "Fit time (seconds)": fit_time
    }

all_model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
selected_model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)

all_results = evaluate_model(all_model, X_train, X_test, y_train, y_test)
selected_results = evaluate_model(
    selected_model, X_train[tree_selected], X_test[tree_selected], y_train, y_test
)

performance = pd.DataFrame(
    [all_results, selected_results],
    index=["All 12 monthly features", "Selected 6 features"]
)
print("\\nModel performance comparison:")
display(performance.round(4))

print("\\nConclusion:")
print("Feature selection was performed using correlation, mutual information, RFE and Random Forest importance.")
print("The same Random Forest regression model was evaluated with all monthly predictors and the top six tree-ranked predictors.")
print("Compare R2, MAE, RMSE and fit time above to judge the trade-off between predictive performance and complexity.")
