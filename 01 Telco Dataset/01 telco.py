# -*- coding: utf-8 -*-
"""
SHAP PARA 4 MODELOS – SMOTE só no treino
Rode como .py: python telco_shap.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend não interativo
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =============================================================================
# Helpers
# =============================================================================
def to_2d_shap(sv, n_features):
    """
    Normaliza a saída do SHAP para (n_samples, n_features).
    Aceita lista por classe, array 2D, ou 3D em formatos comuns.
    """
    # Lista por classe (sklearn árvores costuma retornar)
    if isinstance(sv, list):
        return sv[1] if len(sv) > 1 else sv[0]

    sv = np.asarray(sv)

    # Já é 2D com #features correto?
    if sv.ndim == 2 and sv.shape[1] == n_features:
        return sv

    # Formatos 3D mais comuns:
    # (n_samples, n_classes, n_features)
    if sv.ndim == 3 and sv.shape[2] == n_features and sv.shape[1] in (2, 3):
        cls = 1 if sv.shape[1] > 1 else 0
        return sv[:, cls, :]

    # (n_classes, n_samples, n_features)
    if sv.ndim == 3 and sv.shape[2] == n_features and sv.shape[0] in (2, 3):
        cls = 1 if sv.shape[0] > 1 else 0
        return sv[cls, :, :]

    # (n_samples, n_features, n_classes)
    if sv.ndim == 3 and sv.shape[1] == n_features and sv.shape[2] in (2, 3):
        cls = 1 if sv.shape[2] > 1 else 0
        return sv[:, :, cls]

    # Último recurso: mover eixo de features p/ fim e selecionar o eixo de classe
    for axis in range(sv.ndim):
        if sv.shape[axis] == n_features:
            moved = np.moveaxis(sv, axis, -1)  # features no último eixo
            # escolher o primeiro eixo (antes do último) que pareça ser classes
            for ax in range(moved.ndim - 1):
                if moved.shape[ax] in (2, 3):
                    cls = 1 if moved.shape[ax] > 1 else 0
                    slicer = [slice(None)] * moved.ndim
                    slicer[ax] = cls
                    out = moved[tuple(slicer)]
                    return out.reshape(out.shape[0], n_features)
            return moved.reshape(-1, n_features)

    raise ValueError(f"Formato SHAP inesperado: shape={sv.shape}, n_features={n_features}")

# =============================================================================
# 1) Dados e pré-processamento
# =============================================================================
df = pd.read_csv("01 WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# X original (numérico) para usar nos gráficos
X_original = df.drop('Churn', axis=1).copy()
for col in X_original.select_dtypes(include=['object']).columns:
    X_original[col] = LabelEncoder().fit_transform(X_original[col].astype(str))

# Conjunto de treino
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_test_original = X_original.loc[X_test.index]  # mantém alinhamento de índices

n_features = X_test_original.shape[1]

# =============================================================================
# 2) Modelos
# =============================================================================
models = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        # garantir base_score numérico (não resolve o bug do SHAP, mas é saudável)
        base_score=0.5
    )
}

param_grids = {
    "Regressão Logística": {'model__C': [1], 'model__class_weight': ['balanced']},
    "Árvore de Decisão": {'model__max_depth': [5]},
    "Random Forest": {'model__n_estimators': [100], 'model__max_depth': [10], 'model__class_weight': ['balanced']},
    "XGBoost": {'model__n_estimators': [100], 'model__max_depth': [6], 'model__learning_rate': [0.1]}
}

# =============================================================================
# 3) Treino + SHAP
# =============================================================================
for name, model in models.items():
    print(f"\n--- {name} ---")

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=3,
        scoring=make_scorer(f1_score),
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    trained_model = best_model.named_steps['model']

    # ========== Explainers ==========
    if name == "Regressão Logística":
        # Estável para lineares
        explainer = shap.LinearExplainer(trained_model, X_train, feature_perturbation="interventional")
        sv = explainer.shap_values(X_test_original)  # tende a ser 2D
        shap_values = to_2d_shap(sv, n_features)
        ev = explainer.expected_value
        base_value = ev[1] if np.ndim(ev) and np.size(ev) > 1 else (ev[0] if np.ndim(ev) else ev)

    elif name in ("Árvore de Decisão", "Random Forest"):
        # TreeExplainer funciona bem
        explainer = shap.TreeExplainer(trained_model)
        sv = explainer.shap_values(X_test_original)    # lista, 2D ou 3D
        shap_values = to_2d_shap(sv, n_features)
        ev = explainer.expected_value
        if isinstance(ev, list) or (hasattr(ev, "__len__") and not np.isscalar(ev)):
            base_value = ev[1] if len(ev) > 1 else ev[0]
        else:
            base_value = ev

    else:  # XGBoost
        # >>> Contorno do bug SHAP x XGBoost (base_score '[5E-1]'):
        # Use o explainer por PERMUTAÇÃO. É mais lento, porém robusto.
        # Para manter a interpretação da classe positiva, passamos predict_proba.
        explainer = shap.Explainer(trained_model.predict_proba, X_train, algorithm="permutation")
        shap_obj = explainer(X_test_original)
        sv = shap_obj.values
        # permutation costuma retornar (n_samples, n_classes, n_features)
        shap_values = to_2d_shap(sv, n_features)
        # base_value para força: média da prob. positiva no treino
        base_value = trained_model.predict_proba(X_train)[:, 1].mean()

    # Segurança: garantir formato correto
    assert shap_values.ndim == 2 and shap_values.shape[1] == n_features, \
        f"n_features diferentes: {shap_values.shape[1]} vs {n_features}"

    # ========== Plots ==========
    # SUMMARY
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_original, show=False)
    plt.title(f"SHAP Summary - {name}")
    plt.tight_layout()
    plt.savefig(f"shap_summary_{name.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # DEPENDENCE
    if 'tenure' in X_test_original.columns:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot('tenure', shap_values, X_test_original, show=False)
        plt.title(f"SHAP Dependence: tenure - {name}")
        plt.tight_layout()
        plt.savefig(f"shap_dependence_tenure_{name.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # FORCE (uma única amostra — usa base_value escalar)
    i = 10
    if i < len(X_test_original):
        force_plot = shap.force_plot(base_value, shap_values[i:i+1], X_test_original.iloc[i:i+1], show=False)
        shap.save_html(f"force_plot_{name.replace(' ', '_').lower()}.html", force_plot)

    print(f"  → {name} concluído")

print("\nTODOS OS ARQUIVOS DE SHAP GERADOS SEM ERRO!")
