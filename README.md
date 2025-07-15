# Adult Census Income Prediction Project

## Overview

This project uses data extracted from the 1994 Census Bureau database by Ronny Kohavi and Barry Becker. The dataset is used to predict whether a person’s income exceeds \$50,000 per year (>50K) or not (<=50K), based on demographic and economic features.

* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income/data)
* **Rows & Features**: 32,561 rows and 15 features
* **Key Features**: `age`, `workclass`, `fnlwgt`, `education`, `education.num`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`
* **Target**: `income`

---

## Repository Structure

```
adult_income/                  # Raw dataset file  
adult_income_clean/                # Cleaned dataset file  
DataWorkflow_Adult_Census_Income/ # Jupyter notebooks for preprocessing, modeling, evaluation  
requirements.txt                   # Python dependencies  
README.md                          # Project overview and instructions  
Adult_income_prediction.xls        # Human predictions from our team (user study)
```

---

##  Data Preprocessing Steps

1. Removed duplicate rows
2. Renamed columns:

   * `fnlwgt` → `population.weight`
   * `education.num` → `years.of.education`
3. Standardized incomplete country names:

   * `'South'` → `'South-Korea'`
   * `'Hong'` → `'Hong-Kong'`
4. Randomly modified and corrected entries in `sex` column (e.g., `'F'` → `'Female'`)
5. Removed rows with unknown country (`'?'`)
6. Imputed missing values in `workclass` and `occupation` using **Decision Tree Classifier**
7. Applied `log1p()` transformation on skewed numerical variables
8. Applied **One-Hot Encoding** to categorical columns

---

## ⚙️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ggoel3/DS_Adult_Income.git
cd DS_Adult_Income
```

### 2. Set Up the Environment

Install required libraries:

```bash
pip install -r requirements.txt
```

### 3.  Run Preprocessing and Modeling

####  Data Preprocessing Steps

1. **Remove duplicate rows**
   Eliminate repeated records to avoid data leakage or bias.

2. **Rename columns for clarity**

   * `'fnlwgt'` → `'population.weight'`
   * `'education.num'` → `'years.of.education'`

3. **Standardize country names**
   Replace incomplete or ambiguous entries for better readability:

   * `'South'` → `'South-Korea'`
   * `'Hong'` → `'Hong-Kong'`

4. **Correct inconsistent `sex` values**
   Randomly modified entries like `'F'` and `'M'` are normalized back to `'Female'` and `'Male'`.

5. **Remove unknown country entries**
   Drop rows where `'native.country' == '?'`.

6. **Impute missing values**
   Use a **Decision Tree Classifier** to fill missing values in:

   * `'workclass'`
   * `'occupation'`

7. **Transform skewed numerical features**
   Apply a `log1p` (log(x + 1)) transformation to reduce the skew in:

   * `capital.gain`
   * `capital.loss`
   * `population.weight`

8. **Encode categorical variables**
   Apply **One-Hot Encoding** to all categorical columns (excluding the target).

---

##  Experimental Setup and Key Findings

* **Split**: Data split into **80/20 train-test** with **stratification**
* **Pipeline**: Used `Pipeline` to combine:

  * `StandardScaler` for normalization
  * `SMOTE` to oversample the minority class
  * Logistic Regression or MLPClassifier
* **Cross-validation**: 4-fold Stratified CV with `GridSearchCV`
* **Scoring Metric**: Optimized for **F1 Score** due to class imbalance
* **Target Variable Encoding**: `<=50K` → 0, `>50K` → 1

### Results Summary

| Model                | Accuracy | F1 (Class 1) | Recall (Class 1) |
| -------------------- | -------- | ------------ | ---------------- |
| **Logistic Reg.**    | 0.8068   | 0.6781       | 0.8454           |
| **Neural Network**   | 0.8149   | 0.6722       | 0.7882           |
| **Baseline**         | 0.7592   | 0.0000       | 0.0000           |
| **Human Prediction** | 0.6263   | 0.6105       | 0.6744           |

* The **Neural Network** slightly outperforms Logistic Regression in **recall and accuracy**.
* **Logistic Regression** has a slightly **higher F1** on Class 1 (income >50K), making it more interpretable with comparable results.
* **Baseline model** fails to capture high-income class due to predicting only the majority class.
* **Human predictions**, while reasonable, were less consistent and performed worse than both ML models.

---

##  Challenges and Literature References

###  Major Challenges

* **Missing Values**: `workclass`, `occupation`, and `native.country` contained missing or ambiguous values ('?'), requiring imputation using Decision Tree models.
* **Class Imbalance**: High-income class (`>50K`) was underrepresented (\~25% of samples), leading to biased accuracy. We used **SMOTE** for balancing.
* **Categorical Encoding**: Many features like `workclass` and `education` needed One-Hot Encoding, resulting in high dimensionality.
* **Outliers and Skewness**: Columns like `capital.gain` and `population.weight` were heavily skewed; we addressed this using `log1p()` transformation.
* **Human predictions** showed that even educated guesses tend to overfit to visible demographic signals, lacking statistical generalization.

###  Related Work

* Kohavi (1996) used this dataset and achieved \~85% accuracy using **NBTree** (a hybrid of Naive Bayes and Decision Tree). However, F1 and recall were **not reported** in early works.
* Our approach aligns with modern ML practices:

  * Using **stratified sampling**, **feature scaling**, **class balancing**, and **evaluation beyond accuracy**
  * Avoiding misleading metrics by including **recall, precision, F1** per class

---


## Additional Information

### Steps for Logistic Regression and Neural Network Model Implementation

1. **Split data into features and target**  
   Identify the target variable (`income`) and separate it from predictor features.

2. **Train/Test split**  
   Divide the data into training and test sets, using stratified sampling to maintain the class distribution.

3. **Define pipeline**  
   Combine the following steps into a single pipeline:  
   - Scaler (e.g., `StandardScaler`) for numerical feature normalization  
   - SMOTE for handling class imbalance in the training set  
   - Classifier (either Logistic Regression or Neural Network)

4. **Define hyperparameters to tune**  
   Select candidate hyperparameters for grid search (e.g., regularization strength, hidden layer sizes).

5. **Set up cross-validation strategy**  
   Use 4-fold stratified cross-validation to ensure balanced distribution of classes across folds.

6. **Hyperparameter tuning with GridSearchCV**  
   Perform exhaustive search over the hyperparameter grid, combining cross-validation and scoring.

7. **Fit pipeline to training data**  
   Train models and select the best configuration based on cross-validation scores.

8. **Report best hyperparameters & scores**  
   Display the optimal hyperparameter values and corresponding validation performance.

9. **Evaluate on test set**  
   Assess the final model on unseen test data using multiple metrics (accuracy, F1-score, confusion matrix, etc.).
