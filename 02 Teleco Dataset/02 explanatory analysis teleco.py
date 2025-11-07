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


df = pd.read_csv('02churn-bigml-80.csv')



#Customer service calls, Total day charge, Total day minutes


plt.figure(figsize=(6,5))
sns.countplot(x='Churn', data=df, palette='pastel') 
plt.title('Customer Distribution: Churn vs. Non-Churn')
plt.xlabel('Churn')
plt.ylabel('Number of Costumers')
plt.show()

#Customer service calls
sns.boxplot(x='Churn', y='Customer service calls', data=df)
plt.title('Customer service calls x Churn')
plt.show()

#Total day charge
sns.boxplot(x='Churn', y='Total day charge', data=df)
plt.title('Total day charge x Churn')
plt.show()


#Total day minutes
sns.boxplot(x='Churn', y='Total day minutes', data=df)
plt.title('Total day minutes x Churn')
plt.show()

# # Complains
# sns.countplot(x='Complains', hue='Churn', data=df, palette='pastel')
# plt.title('Churn and Complains')
# plt.xlabel('Is there any complains? (0 = No, 1 = Yes)')
# plt.ylabel('Number of Costumers')
# plt.legend(title='Churn', labels=['No', 'Yes'])
# plt.tight_layout()
# plt.savefig('bar_senior_churn.png')
# plt.show()


# # Status
# sns.countplot(x='Status', hue='Churn', data=df, palette='pastel')
# plt.title('Churn and Status')
# plt.xlabel('Did the customer cancel the service? (1 = No, 2 = Yes)')
# plt.ylabel('Number of Costumers')
# plt.legend(title='Churn', labels=['No', 'Yes'])
# plt.tight_layout()
# plt.savefig('bar_senior_churn.png')
# plt.show()