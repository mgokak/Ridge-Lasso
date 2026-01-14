# Algerian Dataset – EDA & Regularized Regression

## Overview

This repository contains Jupyter Notebooks focused on **Exploratory Data Analysis (EDA)** and **regularized regression techniques** (Ridge and Lasso) applied to the **Algerian dataset**. The goal of these notebooks is to understand the data, explore relationships between variables, and build regression models that handle multicollinearity and overfitting.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. Exploratory Data Analysis (EDA)  
4. Ridge Regression  
5. Lasso Regression  
6. Common Evaluation Metrics 

---

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `EDA_Algerian.ipynb`
- `Algerian_RidgeLassoRegression.ipynb`

---

## Exploratory Data Analysis (EDA)

### `EDA_Algerian.ipynb`

Key points:
- Understand feature distributions
- Identify relationships between variables
- Detect missing values and outliers

Common commands:
```python
df.head()
df.describe()
df.isnull().sum()
sns.histplot(df['feature'])
sns.heatmap(df.corr(), annot=True)
```

---

## Ridge Regression

- Uses **L2 regularization**
- Reduces model complexity
- Handles multicollinearity

Formula:
Loss = RSS + λ Σ β²

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

---

## Lasso Regression

- Uses **L1 regularization**
- Can shrink coefficients to zero
- Performs feature selection

Formula:
Loss = RSS + λ Σ |β|

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
```

---

## Common Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score
```

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

## MSE

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
```

## RMSE

```python
import numpy as np
np.sqrt(mean_squared_error(y_test, y_pred))
```

## R-squared (R²)

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

---

## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
DePaul University
