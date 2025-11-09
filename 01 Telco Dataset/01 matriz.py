import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('01 WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})





# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# df = df.dropna(subset=['TotalCharges'])  # remove linhas com TotalCharges vazio
numeric_df = df.select_dtypes(include=[np.number]) # colunas numéricas selecionadas


corr = numeric_df.corr() # isso gera a matriz correlação

# carreguei as 10 mais importantes
corr_target = corr['Churn'].sort_values(ascending=False)
print("Correlação dos atributos com 'Churn':\n")
print(corr_target)

# matriz correlação
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Matriz correlação')
plt.tight_layout()
plt.show()
