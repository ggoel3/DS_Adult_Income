# DS_Adult_Income
## Overview

This project uses data extracted from the 1994 Census Bureau database by Ronny Kohavi and Barry Becker. The dataset is used to predict whether a person’s income exceeds \$50,000 per year (>50K) or not (<=50K), based on census data (demographic and economic features).

* **Source**:  https://www.kaggle.com/datasets/uciml/adult-census-income/data
* **Number of rows & features**: 32,561 Rows and 15 features
* **Key features**:
  `age`, `workclass`, `fnlwgt`, `education`, `education.num`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`
* **Target Feature**: `income`

## **Dataset Description**

| Column Name      | Data Type | Description                                                  |
| ---------------- | --------- | ------------------------------------------------------------ |
| `age`            | int       | Age of the individual                                        |
| `workclass`      | object    | Type of employment (e.g., Private, Self-emp, Government)     |
| `fnlwgt`         | int       | Final weight — proxy indicator for population representation |
| `education`      | object    | Highest level of education attained                          |
| `education.num`  | int       | Number of years of education                                 |
| `marital.status` | object    | Marital status (e.g., Married, Never-married, Divorced)      |
| `occupation`     | object    | Type of occupation (e.g., Tech-support, Sales, etc.)         |
| `relationship`   | object    | Household relationship (e.g., Husband, Not-in-family)        |
| `race`           | object    | Race of the individual                                       |
| `sex`            | object    | Gender of the individual (`Male` or `Female`)                |
| `capital.gain`   | int       | Capital gains in dollars                                     |
| `capital.loss`   | int       | Capital losses in dollars                                    |
| `hours.per.week` | int       | Number of working hours per week                             |
| `native.country` | object    | Country of origin                                            |
| `income`         | object    | Target variable — income category (`<=50K` or `>50K`)        |


## **Repository Structure**

```
adult_income/                  — raw dataset file  
adult_income_clean/                — clean dataset file  
DataWorkflow_Adult_Census_Income/ — Jupyter notebooks with preprocessing, exploratory data analysis, modeling, and evaluation  
requirements.txt                   — environment dependencies  
README.md                          — project overview and instructions  
```

## **Data checkpoints before pre-processing**

* Checked for null values to identify missing data in all columns
* Checking all Data columns unique values to catch inconsistencies or rare values
* Searched for duplicate records
* Verified data types and reviewed basic statistics of numerical features (creating boxplots to identify outliers)

---

## **Data Cleaning Steps**

* Removing duplicate rows
* Renaming the column name `'fnlwgt'` to `'population.weight'` (a better name for clear understanding)
* Renaming the column name `'education.num'` to `'years.of.education'` (a better name for clear understanding)
* Completing the country names for better understanding (`'South'` to `'South-Korea'` and `'Hong'`, `'Hong-Kong'`)
* Randomly modifying the column `'sex'` values like `'Female'` to `'F'` and `'Male'` to `'M'` and then fix it as `'Male'` and `'Female'`
* Keep only rows where the country is not `'?'`
* Filling in missing values for the columns `'workclass'` and `'occupation'` using DecisionTree Classifier
* Apply `log1p` (log(x+1)) to skewed variables
* Apply One hot encoding to the categorical columns


## **Steps for Logistic Regression and Neural Network Model Implementation**

1. **Split data into features and target**
   Identify the target variable (`income`) and separate it from predictor features.

2. **Train/Test split**
   Divide the data into training and test sets, using stratified sampling to maintain the class distribution.

3. **Define pipeline**
   Combine the following steps into a single pipeline:

   * Scaler (e.g., `StandardScaler`) for numerical feature normalization
   * SMOTE for handling class imbalance in the training set
   * Classifier (either Logistic Regression or Neural Network)

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


## **Steps to Run**

1. **Clone the repository**

   ```bash
   git clone <repo>
   cd <repo>
   ```

2. **Set up the environment**

   ```bash
   pip install -r requirements.txt
   ```

3. **Import all required libraries**

4. **Read the dataset CSV file into a pandas DataFrame**

5. **Run all the steps mentioned under ‘Data checkpoints before pre-processing’**

6. **Clean the data using all the steps mentioned under ‘Data checkpoints before pre-processing’**

7. **Apply Logistics and Neural Network Model to evaluate the performance**
   (Follow all the steps mentioned under “Steps for Logistic Regression and Neural Network Model Implementation”)

8. **Evaluate the Baseline model (`strategy='most_frequent'`)**

9. **Compare all three models (Logistic Regression, Neural Network, and Baseline Model)**

## **Experimental Setup and Key Findings**

We used the 1994 Census income dataset filtered to individuals over 16 years old with sufficient income and working hours. Data preprocessing involved cleaning missing values, encoding categorical variables, and scaling numerical features.

The modeling pipeline compared logistic regression and a neural network, using stratified train-validation-test splits and cross-validation for hyperparameter tuning. Both models achieved similar accuracy (\~85%), but the neural network showed slightly better recall for the high-income class. Key metrics included precision, recall, and F1-score, reflecting balanced performance on imbalanced data.

Kohavi (1996) used the Adult dataset and achieved about 85% accuracy using a Naive Bayes / decision tree hybrid (NBTree). The paper does not report precision, recall, or F1-score, as these metrics were not standard practice at the time.


## **Challenges**

A main challenge was handling missing and inconsistent categorical data, especially in `workclass`, `occupation`, and `native.country`. Rare categories required grouping to avoid sparse feature issues. Balancing interpretability (logistic regression) versus predictive power (neural network) was another focus, aligned with literature where tree-based and deep models often outperform linear baselines but require more careful tuning.

Previous studies emphasize the importance of stratified sampling and proper encoding, which informed our preprocessing choices. Addressing class imbalance and outlier detection was also critical, consistent with standard practices in census income prediction research.
