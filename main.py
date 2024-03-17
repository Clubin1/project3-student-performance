import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def residual_plots():
    plt.figure(figsize=(8, 6))
    plt.scatter(reg.predict(X_train), reg.predict(
        X_train) - y_train, c='b', s=40, alpha=0.5)
    plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100, color='r')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()


def distribution_of_residuals():
    plt.figure(figsize=(8, 6))
    sns.histplot(y_test - y_pred, bins=30, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()


def feature_importance_plots():
    coefficients = pd.Series(reg.coef_, index=X.columns)
    coefficients.plot(kind='bar')
    plt.title('Feature Importance Plot')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.show()


def print_results():
    print('Coefficients: \n', reg.coef_)
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))


def partial_regression_plots():
    fig, axs = plt.subplots(1, len(X.columns), figsize=(15, 6))
    for i, column in enumerate(X.columns):
        sns.regplot(x=X[column], y=y, ax=axs[i],
                    scatter_kws={'s': 20, 'alpha': 0.5})
        axs[i].set_title(
            f'Partial Regression Plot\n({column} vs Performance Index)')
        plt.tight_layout()
        plt.show()


df = pd.read_csv('student_data.csv')

df = pd.get_dummies(df, columns=['Extracurricular Activities'])

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

sns.pairplot(df)
plt.suptitle('Pairplot of Variables')
plt.show()

X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

residual_plots()
distribution_of_residuals()
feature_importance_plots()
partial_regression_plots()
print_results()
