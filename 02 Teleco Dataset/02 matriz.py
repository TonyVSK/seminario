import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('02churn-bigml-80.csv')

# Converter Churn para numérico
df['Churn'] = df['Churn'].astype(int)




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
plt.title('correlation matrix')
plt.tight_layout()
plt.show()
