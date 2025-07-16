# DS_Adult_Income
## Experimental Setup

### Environment 

Following software and python libraries should be installed with their respective version numbers:

Git version: 2.39.5 </br>
Python version: 3.13.1 </br>
ipykernel </br>

### Libraries:
pandas version: 2.2.3 </br>
matplotlib version: 3.9.3 </br>
numpy version: 1.26.2 </br>
seaborn version: 0.13.2 </br>
tensorflow version: 2.19.0 </br>
scikit-learn version: 1.5.2 </br>
imbalanced-learn version: 0.13.0 </br>

### Data Overview: 

This project uses data extracted from the 1994 Census Bureau database by Ronny Kohavi and Barry Becker. The dataset is used to predict whether a person’s income exceeds \$50,000 per year (>50K) or not (<=50K), based on census data (demographic and economic features).

* **Source**:  https://www.kaggle.com/datasets/uciml/adult-census-income/data
* **Number of rows & features**: 32,561 Rows and 15 features
* **Key features**:
  `age`, `workclass`, `fnlwgt`, `education`, `education.num`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`
* **Target Feature**: `income`

### Dataset Description

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


### Data Pre-Processing Steps

* Removing duplicate rows
* Renaming the column name `'fnlwgt'` to `'population.weight'` (a better name for clear understanding)
* Renaming the column name `'education.num'` to `'years.of.education'` (a better name for clear understanding)
* Completing the country names for better understanding (`'South'` to `'South-Korea'` and `'Hong'`, `'Hong-Kong'`)
* Randomly modifying the column `'sex'` values like `'Female'` to `'F'` and `'Male'` to `'M'` and then fix it as `'Male'` and `'Female'`
* Keep only rows where the country is not `'?'`
* Filling in missing values for the columns `'workclass'` and `'occupation'` using DecisionTree Classifier
* Apply `log1p` (log(x+1)) to skewed variables
* Apply One hot encoding to the categorical columns

### Models & Hyperparameters

- Logistic Regression
   Hyperparameter: clf__C
   
- Neural Network 
   Hyperparameters: clf__hidden_layer_sizes: [(50,), (100,), (50, 50)],
                     clf__alpha: [0.0001, 0.001, 0.01],  
                     clf__learning_rate_init: [0.001, 0.01] 
                     
- Baseline model 
   Strategy: majority class
     
- Human Prediction
   Prediction of target feature 'income' on 100 randomly picked observations done by 3 group members

### Experimental Procedure

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

10. **Apply the Baseline model**
    Evaluate the baseline model using most_frequent strategy. 
    
11. **Apply Human prediction**
    Select decent number of random rows from data set and try to predict income based on your understanding. 

12. **Comparison of all models**
    Compare all three models (Logistic Regression, Neural Network, and Baseline Model) and also try to compare their performance with human based predictions as well.
    
    
### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

### Reproducibility
  Random seed: 30

### Key Findings

We used the 1994 Census income dataset filtered to individuals over 16 years old with sufficient income and working hours. Data preprocessing involved cleaning missing values, encoding categorical variables, and scaling numerical features.

Kohavi (1996) used the Adult dataset and achieved about 85% accuracy using a Naive Bayes / decision tree hybrid (NBTree). The paper does not report precision, recall, or F1-score, as these metrics were not standard practice at the time.

### Our models Comparison summary:
                 Model                Accuracy  F1 (class 0)  Recall (class 0)   Precision (class 0) 
                 Baseline (Majority)  0.7592        0.8631            1.0000     0.7592  
                 Logistic Regression  0.8068        0.8619            0.7945     0.9419  
                 Neural Network       0.8149        0.8710            0.8234     0.9246  
                 Human Prediction     0.6263        0.6408            0.5893     0.7021  

                 Model                            F1 (class 1)  Recall (class 1) Precision (class 1)  
                 Baseline (Majority).               0.0000            0.0000     0.0000  
                 Logistic Regression                0.6781            0.8454     0.5661  
                 Neural Network                     0.6722            0.7882     0.5860  
                 Human Prediction                   0.6105            0.6744     0.5577  

### Overall Performance:

* The Neural Network achieved the highest accuracy (0.8149) and best balanced F1 score on class 0 (0.8710) compared to other models.

* Logistic Regression performed very competitively:
  Slightly lower accuracy (0.8068) than the neural network.
  Achieved the highest recall for class 1 (0.8454) among all models, making it better at identifying the minority class.

* Human prediction had noticeably lower accuracy (0.6263) and F1 scores, but still performed better than the baseline on minority class prediction:
  F1 (class 1): 0.6105 vs. baseline 0.0000

* Baseline (majority class) predictor:
  Still Better accuracy (0.7592) because it always predicts the majority class.
  Completely fails on minority class: F1 and recall for class 1 are both 0.0000.
---

#  Challenges and Literature References

### Major Challenges

* **Missing Values**: `workclass`, `occupation` contained missing or ambiguous values ('?'), requiring imputation using Decision Tree models.
* **Class Imbalance**: High-income class (`>50K`) was underrepresented (\~25% of samples), leading to biased accuracy. We used **SMOTE** for balancing.
* **Categorical Encoding**: Many features like `workclass` and `education` needed One-Hot Encoding, resulting in high dimensionality.
* **Outliers and Skewness**: Columns like `capital.gain` and `population.weight` were heavily skewed; we addressed this using `log1p()` transformation.
* **Human predictions** showed that even educated guesses tend to overfit to visible demographic signals, lacking statistical generalization.

### Related Work

* Kohavi (1996) used this dataset and achieved \~85% accuracy using **NBTree** (a hybrid of Naive Bayes and Decision Tree). However, F1 and recall were **not reported** in early works.

# Navigating Repository

### Repository Structure

```
adult_income/                      — raw dataset file  
adult_income_clean/                — clean dataset file  
DataWorkflow_Adult_Census_Income/  — Jupyter notebooks with preprocessing, exploratory data analysis, modeling, and evaluation  
requirements.txt                   — environment dependencies  
README.md                          — project overview and instructions  
```

### Steps to Re-Run the Experiment

1. Clone the repository
   ```bash
   git clone <https://github.com/ggoel3/DS_Adult_Income.git>
   cd <DS_Adult_Income>
   ```
2. Create and activate virtual environment
  ```python -m venv venv
  source venv/bin/activate          # On Linux/Mac
  venv\Scripts\activate             # On Windows
  ```
3. Set up the environment
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook with name 'DataWorkflow_Adult_Census_Income.ipynb'
