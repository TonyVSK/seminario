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


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# reciclei o código a baixo da atividade prática que entregamos no moodle
plt.figure(figsize=(6,5))
sns.countplot(x='Churn', data=df, palette='pastel') 
plt.title('Distribuição de Clientes: Churn vs Não Churn')
plt.xlabel('Churn')
plt.ylabel('Número de Clientes')
plt.show()
