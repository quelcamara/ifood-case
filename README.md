## 📦 iFood Coupon Conversion Prediction
Projeto de Data Science desenvolvido para prever a probabilidade de um cliente converter (utilizar) uma oferta enviada, com o objetivo de otimizar a distribuição de cupons e melhorar o ROI das campanhas promocionais.

## 🎯 Objetivo
O projeto busca desenvolver um modelo preditivo capaz de identificar quais combinações cliente-oferta têm maior propensão de conversão, permitindo ao iFood:

- Reduzir desperdício de cupons;
- Priorizar os clientes com maior potencial de engajamento;
- Tomar decisões baseadas em dados sobre quem abordar e com qual cupom.

## 🔍 Abordagem
A solução foi construída com foco em interpretação e geração de insights para campanhas. O processo foi dividido em três etapas principais:

1. Exploração e Limpeza de Dados
  - Avaliação das bases fornecidas: clientes, ofertas e transações
  - Junção das bases utilizando conhecimento de negócio para construir a jornada do cliente
  - Definição da variável target -- sucesso do cliente para determinada oferta
  - Criação de variáveis como: tempo entre eventos, taxas de recebimento de oferta por canal, frequência de visualização de cupons, etc.

2. Feature Engineering e Modelagem
  - Defasagem de informação para cálculo de features temporais
  - Criação de features como `avg_ticket_15d`, `avg_days_received_view_30d`, `pct_type_discount_15d`, etc.
  - Agregação da base final para nível cliente-oferta
  - Seleção inicial de variáveis a partir do Permutation Importance
  - Otimização de hiperparâmetros com uso do Optuna
  - Modelagem partindo de algoritmo baseados em árvores (LightGBM)
  - Calibração do modelo utilizando o `CalibratedClassifierCV`

3. Análise de Resultados e Estratégias de Negócio
  - Cálculo de métricas de modelo: AUC, Precision, Recall.
  - Verificação da calibração do modelo a partir do plot de curva de calibração
  - Cálculo de métricas de negócio: Precision@k, Recall@k, Lift Curve.
  - Interpretação do impacto das variáveis via SHAP values
  - Análises por decil e segmentação de variáveis
  - Geração de insights

## 📁 Estrutura do Projeto
```
  ifood-case/
  ├── data/
  │   ├── raw/             # Dados originais (input)
  │   └── processed/       # Dados tratados e agregados
  ├── notebooks/
  │   ├── 0_data_exploratory.ipynb     # Análise exploratória
  │   ├── 1_data_processing.ipynb      # Processamento de dados e feature engineering
  │   └── 2_modeling.ipynb             # Feature selection, treinamento e análise do modelo
  ├── presentation/        # Slides finais com insights e storytelling
  ├── src/                 # Código modularizado (preprocessing, hyperopt, model, utils, plots)
  ├── README.md
  └── requirements.txt     # Dependências do projeto
```

## 📊 Métricas-chave
- Lift (Top 20%): O modelo captura mais de 2.63 vezes a taxa de conversão esperada com abordagem aleatória
- Conversão@20%: 0.8764
- Precision@20%: 0.8581
- Recall@20%: 0.5569
- SHAP: Interpretação das principais variáveis de decisão

📈 Insights Relevantes
- As top 20% melhores combinações cliente-oferta associam pelo menos 01 oferta para mais de 40% dos clientes.
- Variáveis como `pct_viewed_offers_15d`, `credit_card_limit`, `avg_ticket_15d` se destacaram como preditoras.
- Políticas de abordagem segmentada por decil de score mostram alto potencial de ganho.

## 🚀 Como Rodar
1. Clone o repositório:
```
git clone https://github.com/seuusuario/ifood-case.git
cd ifood-case
```

2. Crie o ambiente e instale dependências:
```
pip install -r requirements.txt
```
3. Navegue pelos notebooks na ordem:
  - 0_data_exploratory.ipynb
  - 1_data_processing.ipynb
  - 2_modeling.ipynb

## 📌 Requisitos
Veja requirements.txt para bibliotecas necessárias. Principais:
- pandas
- numpy
- scikit-learn
- lightgbm
- shap
- matplotlib / seaborn

## 🧑‍💼 Para Stakeholders
Os slides com storytelling e recomendações estratégicas estão disponíveis em:
```
/presentation/
```

## 📬 Contato
Projeto desenvolvido por [Raquel Câmara](https://www.linkedin.com/in/raquel-camara/)
