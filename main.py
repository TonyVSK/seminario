import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

# Banco de dados arquivo CSV com dados:
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# testar upload do banco:
print(df.head())



##################### ETAPA METODOLOGIA 3.1 : TRARAR PRE-PROCESSAMENTO #############################


df = df.drop(['customerID'], axis=1) # com isso removemos as colunas irrelevantes, como customerID
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna() # Com isso normalizamos para numerico para não dar problema
label_encoders = {}  # Codificar variáveis categóricas
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop('Churn', axis=1) 
y = df['Churn'] # em x e y separamos variáveis preditoras e alvo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # dados em 70 porcento e 30 porcento 
