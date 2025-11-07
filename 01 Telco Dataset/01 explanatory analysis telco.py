import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import warnings


df = pd.read_csv('01 WA_Fn-UseC_-Telco-Customer-Churn.csv')


plt.figure(figsize=(6,5))
sns.countplot(x='Churn', data=df, palette='pastel') 
plt.title('Customer Distribution: Churn vs. Non-Churn')
plt.xlabel('Churn')
plt.ylabel('Number of costumers')
plt.show()


sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges x Churn')
plt.show()



plt.figure(figsize=(7,5))
sns.countplot(x='SeniorCitizen', hue='Churn', data=df, palette='pastel')
plt.title('Churn distribution and elderly customers')
plt.xlabel('Is the customer elderly? (0 = No, 1 = Yes)')
plt.ylabel('Number of Costumers')
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('bar_senior_churn.png')
plt.show()
