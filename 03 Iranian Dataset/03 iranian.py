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
# ITS NECESSARY TO HAVE matplotlib TO RUN THE SHAP AND SEE THE GRAPHICS!!!



# Banco de dados arquivo CSV com dados:
df = pd.read_csv("03 iranian churn dataset.csv")

# testar upload do banco:
print(df.head())



# ##################### ETAPA METODOLOGIA 3.1 : TRATAR PRE-PROCESSAMENTO #############################


df = df.dropna() # Com isso normalizamos para numerico para não dar problema
label_encoders = {}  # Codificar variáveis categóricas
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop('Churn', axis=1) 
y = df['Churn'] # em x e y separamos variáveis preditoras e alvo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # dados em 70 porcento e 30 porcento



# ##################### ETAPA METODOLOGIA 3.2 : TREINAR MODELOS QUE SERÃO USADOS #############################


models = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, base_score=0.5, use_label_encoder=False)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred)}

results_df = pd.DataFrame(results).T
print(results_df)





































##################### ETAPA METODOLOGIA 3.4 : EXPLICABILIDADE #############################
## 
# NOVA API EXPLÍCITA - PRECISEI APLICAR UMA SOLUÇÃO DIFERENTE PARA CONSEGUIR EXECUTAR O CÓDIGO

# Modelo XGBoost treinado
model_xgb = models["Random Forest"]

# Masker explicito para dados, assim o SHAP vai saber lidar com dataframe x_train
masker = shap.maskers.Independent(X_train)
print("INICIANDO... ")
explainer = shap.Explainer(model_xgb.predict_proba, masker)

####
####
####
# FIM API EXPLICITA
####
####
####

print("Calculando SHAP values (pode demorar um pouco)...")
# KernelExplainer => lento
shap_explanation_values = explainer(X_test).values

# O resultado será uma lista [valores_classe_0, valores_classe_1]
# Vamos pegar apenas os da classe 1 (Churn)
shap_values_class_1 = shap_explanation_values[:,:,1]

print("Cálculo concluído.")

# NOTA: A plotagem precisa ser ajustada
print("Plotando gráficos SHAP...")
shap.summary_plot(shap_values_class_1, X_test, plot_type="bar", show=True)
shap.summary_plot(shap_values_class_1, X_test, show=True)

shap.initjs()
i = 10

# O force_plot precisa ser reconstruído, pois a nova API é diferente
# Precisamos do 'base_value' da classe 1
base_value_class_1 = explainer.expected_value[1]
force_plot = shap.force_plot(base_value_class_1, shap_values_class_1[i,:], X_test.iloc[i,:])
shap.save_html("force_plot_tree_new_api.html", force_plot) # Salva como HTML
print(f"Gráfico force_plot (Tree) salvo como 'force_plot_tree_new_api.html'")
