
# Titanic Survival Prediction

## Project Overview

This project is a machine learning classification project based on the Titanic dataset.

The goal is to predict whether a passenger survived or not using passenger information such as gender, passenger class, age group, family size, and cabin availability.

The target variable is:

- `Survived`
  - `0` = Did not survive
  - `1` = Survived

Since the target variable has only two possible outcomes, this is a **binary classification** problem.

---

## Dataset

The dataset consists of two main files:

- `train.csv`: Used for training and validating the machine learning models. This file includes the `Survived` column.
- `test.csv`: Used for final predictions. This file does not include the `Survived` column.
- `gender_submission.csv`: An example submission file showing the required Kaggle format.

---

## Exploratory Data Analysis

Exploratory data analysis was performed to understand how different variables are related to survival.

### Gender

Gender was one of the strongest predictors of survival.

Female passengers had a much higher survival rate than male passengers. Therefore, the `Sex` column was included as an important feature.

The encoding was:

```python
male = 0
female = 1
```

### Passenger Class

Passenger class also showed a clear relationship with survival.

First-class passengers generally had a higher survival rate compared to second and third-class passengers. Therefore, `Pclass` was included as a model feature.

### Family Size

The `SibSp` and `Parch` columns were combined to create a new feature called `FamilySize`.

```python
FamilySize = SibSp + Parch + 1
```

The `+1` represents the passenger themself.

This feature helps describe whether a passenger was traveling alone, with a small family, or with a larger family.

### IsAlone

A binary feature was created from `FamilySize`.

```python
IsAlone = 1 if FamilySize == 1 else 0
```

Although this feature is derived from `FamilySize`, it was kept in the final feature set.

### Cabin Information

The raw `Cabin` column was not used directly because it contains many missing values and individual cabin numbers are too specific.

Instead, a new binary feature was created:

```python
HasCabin = 1 if cabin information exists else 0
```

Passengers with cabin information had a higher survival rate, but this feature should be interpreted carefully because cabin availability is also related to passenger class.

### Age

Exact age values were difficult to interpret directly because there were many unique values.

Therefore, passengers were grouped into age bins:

| AgeBin | Age Group |
|---|---|
| `0` | Child |
| `1` | Teen |
| `2` | Young Adult |
| `3` | Adult |
| `4` | Senior |

Age missing values were filled before creating the `AgeBin` feature.

---

## Features Used

After exploratory data analysis and feature engineering, the following features were selected:

| Feature | Description |
|---|---|
| `Pclass` | Passenger ticket class |
| `Sex` | Passenger gender encoded as numeric values |
| `FamilySize` | Total number of family members on board including the passenger |
| `HasCabin` | Whether the passenger has cabin information |
| `IsAlone` | Whether the passenger was traveling alone |
| `AgeBin` | Age group of the passenger |

Final feature set:

```python
features = [
    "Pclass",
    "Sex",
    "FamilySize",
    "HasCabin",
    "IsAlone",
    "AgeBin"
]
```

---

## Data Preprocessing

The preprocessing steps included:

1. Handling missing values
2. Encoding categorical variables
3. Creating new features
4. Selecting final model features

### Missing Value Handling

The `Age` column had missing values. These values were filled using median imputation.

```python
from sklearn.impute import SimpleImputer

age_imputer = SimpleImputer(strategy="median")
dataset[["Age"]] = age_imputer.fit_transform(dataset[["Age"]])
```

The `Cabin` column was not filled directly because missing cabin values were meaningful for this project. Instead, the `HasCabin` feature was created.

### Encoding

The `Sex` column was converted into numeric values.

```python
dataset["Sex"] = dataset["Sex"].map({
    "male": 0,
    "female": 1
})
```

---

## Feature Engineering

The following features were created:

```python
dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

dataset["IsAlone"] = (dataset["FamilySize"] == 1).astype(int)

dataset["HasCabin"] = dataset["Cabin"].notnull().astype(int)

dataset["AgeBin"] = pd.cut(
    dataset["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[0, 1, 2, 3, 4],
    include_lowest=True
).astype(int)
```

The final input and target variables were prepared as:

```python
X = dataset[features]
y = dataset["Survived"]
```

---

## Models Tested

Several classification algorithms were tested and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors
- Support Vector Classifier
- Kernel SVM

The best validation performance was achieved by the default `SVC()` model.

---

## Final Model

The final selected model was:

```python
from sklearn.svm import SVC

final_model = SVC()
```

The model was trained using:

```python
final_model.fit(X_train, y_train)
```

---

## Validation Results

The final SVC model achieved the following validation accuracy:

```text
Accuracy: 0.8324
```

Classification report:

```text
              precision    recall  f1-score   support

           0       0.84      0.90      0.87       110
           1       0.82      0.72      0.77        69

    accuracy                           0.83       179
   macro avg       0.83      0.81      0.82       179
weighted avg       0.83      0.83      0.83       179
```

Confusion matrix:

```text
[[99 11]
 [19 50]]
```

The model performed well overall and showed a balanced performance for both survived and non-survived passengers.


---

## Test Prediction and Submission

Since `test.csv` does not contain the `Survived` column, the final model was used to predict survival values for the test dataset.

The same preprocessing and feature engineering steps were applied to the test dataset.

One important point is that the test feature matrix must have the same column order as the training feature matrix:

```python
X_test = test_dataset[X.columns]
```

The final predictions were generated using:

```python
test_predictions = final_model.predict(X_test)
```

The submission file was created as:

```python
submission = pd.DataFrame({
    "PassengerId": test_dataset["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("submission.csv", index=False)
```

The final submission file contains two columns:

| PassengerId | Survived |
|---|---|
| Passenger ID from test.csv | Predicted survival value |

---

## Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── README.md
├── titanic_survival_prediction.ipynb
├── train.csv
├── test.csv
└── submission.csv
```

---

## Conclusion

This project demonstrates a complete machine learning classification workflow using the Titanic dataset.

The main steps included:

- Data exploration
- Missing value handling
- Feature engineering
- Encoding
- Model comparison
- Validation
- Test prediction
- Submission file creation

The final `SVC()` model achieved approximately **83.2% validation accuracy**.

The most useful features for survival prediction were gender, passenger class, age group, family size, and cabin availability.
