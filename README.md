# Gym Calories ML Analysis

Machine learning analysis of gym workout data, including calorie prediction and classification, exploratory analysis, and workout clustering with scikit-learn.

## Setup

The scripts are standalone analyses that use the shared cleaned CSV in the project root. Use Python 3.10 or later and install the listed data, plotting, and machine-learning packages:

```bash
python -m pip install -r requirements.txt
```

Run each script from the project directory so its relative dataset path resolves correctly. For example, the EDA script prints dataset checks and opens its plots:

```bash
python main_EDA.py
```

## Dataset

`gym_dataset_cleaned.csv` is the shared input for the analyses. It includes workout and participant attributes such as session duration, calories burned, age, weight, average BPM, workout type, and workout frequency. The EDA script reports shape, columns, data types, missing values, duplicates, and descriptive statistics; it then removes duplicate rows, saves the cleaned CSV, and plots the calorie distribution and numeric-feature correlation heatmap. Dataset provenance is not documented in the project yet.

## Analyses

Each script loads the shared CSV, performs one analysis, and prints its results to the terminal. Scripts with visualizations also open a Matplotlib or Seaborn plot window.

| Script | Analysis and output |
| --- | --- |
| `main_EDA.py` | Prints data inspection and cleaning checks, then displays a calorie histogram and a correlation heatmap. |
| `Linear_Regression.py` | Fits calories burned from session duration; prints MAE, MSE, RMSE, and R-squared, then plots test observations and the regression line. |
| `MultipleLinearRegression.py` | Uses five workout and participant features to predict calories; prints regression metrics and plots actual versus predicted values. |
| `PolynomialRegression.py` | Compares a degree-3 polynomial with a linear model for session duration; prints polynomial regression metrics and plots both fitted curves with test data. |
| `LogisticRegression.py` | Classifies calories as above or below the dataset median using five features; prints accuracy, log loss, ROC-AUC, a classification report, and confusion matrix, then plots the matrix. |
| `KNN.py` | Min-Max normalizes four features for median-based high/low classification; prints data summaries, accuracy, and confusion matrix, then plots the matrix. |
| `SVM.py` | Trains a linear SVM using session duration and average BPM; prints accuracy, precision, recall, and F1, then displays the decision boundary and labeled observations. |
| `Decision_Tree.py` | Trains a depth-limited entropy tree on five features; prints classification metrics, confusion matrix, and feature importances, then displays the tree. |
| `K_Means_Clustering.py` | Standardizes session duration and calories before fitting three clusters; prints cluster counts and centers in original units, then plots records and centers. |

## Results

The table is intentionally unfilled: run the scripts against the final dataset and record their printed values here. Keep the train/test settings and dataset version consistent when comparing supervised models; K-Means is summarized with cluster counts and centers rather than classification scores.

| Model | Metrics to report | Result |
| --- | --- | --- |
| Simple linear regression | MAE, MSE, RMSE, R-squared | TODO |
| Multiple linear regression | MAE, MSE, RMSE, R-squared | TODO |
| Polynomial regression | MAE, MSE, RMSE, R-squared | TODO |
| Logistic regression | Accuracy, log loss, ROC-AUC | TODO |
| KNN | Accuracy | TODO |
| Linear SVM | Accuracy, precision, recall, F1 | TODO |
| Decision tree | Accuracy, log loss, ROC-AUC | TODO |
| K-Means | Cluster counts and centers | TODO |

## Roadmap

These are follow-up items for making the analysis easier to interpret, reproduce, and extend:

- Add measured results and compare model performance.
- Document dataset provenance and validation methodology.
- Add repeatable tests and command-line configuration.

## Author

Add the project author's preferred name and public profile link here before publishing the README as complete.