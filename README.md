# Projeto1_PANDAS-PrevisaoSobreviventesTitanic

# Previsão de Sobreviventes do Titanic

## 📄 Descrição do Projeto

Este projeto utiliza o famoso dataset do Titanic para construir e avaliar modelos de Machine Learning com o objetivo de prever se um passageiro sobreviveu ou não ao desastre. Foram explorados, pré-processados e modelados os dados utilizando as bibliotecas Scikit-learn e Pandas.

## 🎯 Objetivo

O objetivo principal é comparar o desempenho de dois modelos de classificação distintos:
1.  **Regressão Logística**: Um modelo linear, simples e interpretável.
2.  **Random Forest**: Um modelo de ensemble, mais complexo e robusto.

## 🛠️ Ferramentas e Bibliotecas Utilizadas

- Python 3.x
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

Antes de tudo, certifique-se de ter o Python 3 instalado.

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/SEU_USUARIO/previsao-sobreviventes-titanic.git](https://github.com/SEU_USUARIO/previsao-sobreviventes-titanic.git)
    ```
2.  Navegue até o diretório do projeto:
    ```bash
    cd previsao-sobreviventes-titanic
    ```
3.  Crie e ative um ambiente virtual (altamente recomendado):
    ```bash
    python -m venv venv
    # No Linux/macOS:
    source venv/bin/activate
    # No Windows:
    venv\Scripts\activate
    ```
4.  Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```

Após a configuração inicial, você tem duas opções para rodar o projeto:

---

### Opção 1: Análise Interativa (Jupyter Notebook)

Esta é a melhor opção para explorar o código passo a passo, visualizar as saídas e entender a análise.

1.  Inicie o Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  No seu navegador, abra o arquivo `analise_titanic.ipynb`.
3.  Execute as células de código sequencialmente.

---

### Opção 2: Execução Direta (Script Python)

Esta opção roda todo o processo de uma só vez, do início ao fim, diretamente do seu terminal. É ideal para ver o resultado final rapidamente.

1.  Certifique-se de que você está no terminal com o ambiente virtual ativado e as dependências instaladas.

2.  Execute o script `analise_titanic.py` com o seguinte comando:
    ```bash
    python analise_titanic.py
    ```

3.  **O que esperar:**
    * As informações e os relatórios de classificação serão impressos diretamente no seu terminal.
    * Duas janelas de gráfico (as Matrizes de Confusão) irão aparecer, uma de cada vez. **Você precisará fechar a primeira janela do gráfico para que o script continue e mostre a segunda.**

## 📈 Resultados e Conclusão

Ambos os modelos foram treinados e avaliados utilizando métricas como Acurácia, Precisão, Recall e F1-Score.

| Modelo | Acurácia | Precisão (sobreviveu) | Recall (sobreviveu) |
| ------------------- | -------- | --------------------- | ------------------- |
| Regressão Logística | ~80%     | ~0.76                 | ~0.78               |
| **Random Forest** | **~82%** | **~0.78** | **~0.78** |

**Conclusão**: O modelo **Random Forest apresentou um desempenho superior**. Isso se deve à sua capacidade de capturar relações não-lineares e interações complexas entre as features, algo que a Regressão Logística, por ser um modelo linear, não consegue fazer com a mesma eficácia.

---
*Desenvolvido por Heitor Giometti*
