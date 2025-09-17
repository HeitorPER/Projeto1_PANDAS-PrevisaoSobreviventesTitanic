# Importação das bibliotecas essenciais
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importação de componentes do Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o dataset
# Certifique-se de que o arquivo 'train.csv' está na mesma pasta do seu notebook
df = pd.read_csv('train.csv')

# Visualizar as primeiras linhas e informações gerais
print(df.head())
print("\nInformações do DataFrame:")
df.info()

# --- Tratamento de Dados Faltantes ---

# Idade (Age): Preencher com a mediana, que é menos sensível a outliers
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarque (Embarked): Preencher com a moda (o valor mais comum)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Cabine (Cabin): Possui muitos valores nulos. A estratégia mais simples é removê-la.
df.drop('Cabin', axis=1, inplace=True)

# Verificar se ainda há dados nulos
print("\nDados nulos após tratamento:")
print(df.isnull().sum())


# --- Engenharia e Seleção de Features ---

# Remover colunas que não agregam valor preditivo para um modelo simples
# Name e Ticket são muito únicos, PassengerId é apenas um identificador
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


# --- Preparação para o Modelo ---

# 1. Definir as features (X) e o alvo (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# 2. Identificar colunas categóricas e numéricas
categorical_features = ['Pclass', 'Sex', 'Embarked']
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

# 3. Criar um pipeline de pré-processamento
#    - Para dados numéricos: Padronização (coloca na mesma escala)
#    - Para dados categóricos: One-Hot Encoding (transforma em colunas binárias)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 4. Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modelo 1: Regressão Logística ---

# Criar o pipeline com o pré-processador e o classificador
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42))])

# Treinar o modelo
print("\nTreinando o modelo de Regressão Logística...")
lr_pipeline.fit(X_train, y_train)


# --- Modelo 2: Random Forest ---

# Criar o pipeline com o pré-processador e o classificador
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Treinar o modelo
print("Treinando o modelo de Random Forest...")
rf_pipeline.fit(X_train, y_train)

# Fazer previsões com ambos os modelos
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)

# --- Avaliação da Regressão Logística ---
print("\n--- Resultados da Regressão Logística ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_lr))

# Matriz de Confusão
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Regressão Logística')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.show()


# --- Avaliação do Random Forest ---
print("\n--- Resultados do Random Forest ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_rf))

# Matriz de Confusão
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges')
plt.title('Matriz de Confusão - Random Forest')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.show()
