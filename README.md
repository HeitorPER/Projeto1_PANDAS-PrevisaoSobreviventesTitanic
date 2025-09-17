# Projeto1_PANDAS-PrevisaoSobreviventesTitanic

# Previsão de Sobreviventes do Titanic

## 📄 Descrição do Projeto

Este projeto utiliza o famoso dataset do Titanic para construir e avaliar modelos de Machine Learning com o objetivo de prever se um passageiro sobreviveu ou não ao desastre. Foram explorados, pré-processados e modelados os dados utilizando as bibliotecas Scikit-learn e Pandas.

## 🎯 Objetivo

O objetivo principal é comparar o desempenho de dois modelos de classificação distintos:
1.  **Regressão Logística**: Um modelo linear, simples e interpretável.
2.  **Random Forest**: Um modelo de ensemble, mais complexo e robusto.

## 🛠️ Ferramentas e Bibliotecas Utilizadas

- Python 3.0
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## 📂 Estrutura dos Arquivos

```
.
├── analise_titanic.ipynb   # Notebook com todo o processo de análise e modelagem
├── train.csv               # Dataset de treino original do Kaggle
├── requirements.txt        # Lista de dependências para reprodução do ambiente
└── README.md               # Documentação do projeto
```

## ⚙️ Como Executar o Projeto

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/SEU_USUARIO/previsao-sobreviventes-titanic.git](https://github.com/SEU_USUARIO/previsao-sobreviventes-titanic.git)
    ```
2.  Navegue até o diretório do projeto:
    ```bash
    cd previsao-sobreviventes-titanic
    ```
3.  Crie um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```
4.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
5.  Abra o Jupyter Notebook:
    ```bash
    jupyter notebook analise_titanic.ipynb
    ```

## 📈 Resultados e Conclusão

Ambos os modelos foram treinados e avaliados utilizando métricas como Acurácia, Precisão, Recall e F1-Score.

| Modelo | Acurácia | Precisão (sobreviveu) | Recall (sobreviveu) |
| ------------------- | -------- | --------------------- | ------------------- |
| Regressão Logística | ~80%     | ~0.76                 | ~0.78               |
| **Random Forest** | **~82%** | **~0.78** | **~0.78** |

**Conclusão**: O modelo **Random Forest apresentou um desempenho superior**. Isso se deve à sua capacidade de capturar relações não-lineares e interações complexas entre as features, algo que a Regressão Logística, por ser um modelo linear, não consegue fazer com a mesma eficácia.

---
*Desenvolvido por Heitor Giometti*
