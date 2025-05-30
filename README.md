# Classificador de Autores com BERTimbau

Este projeto utiliza um modelo BERT ([BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased)) ajustado (fine-tuned) para classificar mensagens entre 5 autores distintos, a partir do conteúdo textual.

## 📊 Relatório de Desempenho

| Classe            | Precision | Recall | F1-Score | Suporte |
|-------------------|-----------|--------|----------|---------|
| Daniel Unirio     | 0.60      | 0.57   | 0.58     | 500     |
| Diogo             | 0.53      | 0.57   | 0.55     | 500     |
| Evangelista       | 0.61      | 0.61   | 0.61     | 500     |
| João Maurício     | 0.64      | 0.68   | 0.66     | 500     |
| Matheus Unirio    | 0.65      | 0.60   | 0.62     | 500     |
| **Acurácia Total**|           |        | **0.60** | 2500    |
| **Média Macro**   | 0.61      | 0.60   | 0.61     | 2500    |
| **Média Ponderada**| 0.61     | 0.60   | 0.61     | 2500    |

## 📌 Visualizações

### 🎯 Matriz de Confusão

> ![Matriz de Confusão](images/matriz.png)

### 🔎 Dispersão dos Embeddings `[CLS]` (t-SNE)

> ![Gráfico de Dispersão](images/scatter.png)

## 🛠️ Tecnologias

- Python
- PyTorch
- Transformers (Hugging Face)
- [chat-miner](https://github.com/joweich/chat-miner)
- Scikit-learn
- Matplotlib / Seaborn
