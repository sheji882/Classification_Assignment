# Breast Cancer Classification – Supervised Learning Assignment

##  Objective

The goal of this assignment is to evaluate supervised learning techniques by applying five classification algorithms to the Breast Cancer dataset available in the `sklearn` library. The focus is on model implementation, preprocessing, and performance comparison.

##  Dataset

- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Description**: This dataset contains features computed from digitized images of breast masses and a label indicating whether the tumor is malignant or benign.


## Technologies Used

- Python  
- Jupyter Notebook  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib / Seaborn (for visualizations)


##  Steps Completed

### 1.  Loading and Preprocessing (2 marks)

- Loaded the Breast Cancer dataset using `sklearn.datasets`.
- Converted the data into a pandas DataFrame.
- Checked for missing values (none found).
- Applied **StandardScaler** for feature scaling to normalize input features.

*Justification*: Feature scaling is necessary for algorithms like k-NN and SVM, which are sensitive to the scale of input data.

---

### 2.  Classification Models Implemented (5 marks)

| Algorithm               | Description                                                                                   | Suitability                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Logistic Regression** | Linear classifier that models the probability of a binary outcome using a logistic function. | Great for binary classification problems with interpretable coefficients.   |
| **Decision Tree**       | Uses a tree-like structure to model decisions and outcomes.                                   | Handles both categorical and numerical data well; interpretable.             |
| **Random Forest**       | Ensemble of decision trees using bagging for better generalization.                          | Reduces overfitting and improves accuracy on tabular data.                   |
| **SVM (Support Vector)**| Finds the optimal hyperplane that separates classes in feature space.                        | Effective in high-dimensional spaces; requires scaling.                      |
| **k-Nearest Neighbors** | Classifies based on the majority class of k nearest neighbors.                               | Simple and effective; sensitive to feature scaling and value of k.           |

---

### 3.  Model Comparison (2 marks)

All models were evaluated using accuracy score and classification report (precision, recall, F1-score).

| Model                  | Accuracy Score |
|-----------------------|----------------|
| Logistic Regression   | 97%            |
| Decision Tree         | 93%            |
| Random Forest         | 98% ✅         |
| SVM                   | 96%            |
| k-Nearest Neighbors   | 95%            |

 **Best Performer**: Random Forest Classifier  
 **Least Performer**: Decision Tree (slightly lower due to overfitting on small splits)

