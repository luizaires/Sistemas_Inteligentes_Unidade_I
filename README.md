# Problema 1 – Regressão Linear para Previsão da Produção de Energia

## Objetivo
Desenvolver um modelo de regressão linear múltipla para prever a produção de energia elétrica de uma usina de ciclo combinado, utilizando variáveis ambientais coletadas por sensores.

## Dataset
Foi utilizado o dataset **UCI Combined Cycle Power Plant (CCPP)**, contendo 9.568 amostras coletadas ao longo de 6 anos.

### Variáveis do dataset

| Variável | Descrição | Unidade |
|---|---|---|
| AT | Temperatura ambiente | °C |
| V | Vácuo de escape | cm Hg |
| AP | Pressão atmosférica | mbar |
| RH | Umidade relativa do ar | % |
| PE | Produção líquida de energia (variável alvo) | MW |

**Fonte:** [UCI Combined Cycle Power Plant Dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant).

## Tarefas realizadas

1. **Análise de correlação:** avaliação da relação entre as variáveis independentes (AT, V, AP e RH) e a variável alvo (PE), identificando as variáveis com maior influência na produção de energia.

2. **Análise de multicolinearidade:** investigação da correlação entre as variáveis independentes.

3. **Pré-processamento:** tratamento de valores ausentes, normalização ou padronização das variáveis e divisão dos dados em conjuntos de treinamento (80%) e teste (20%).

4. **Construção do modelo:** implementação de um modelo de regressão linear múltipla utilizando AT, V, AP e RH para prever PE.

5. **Avaliação do modelo:** análise do desempenho por meio das métricas:
   - R² (coeficiente de determinação);
   - RMSE (raiz do erro quadrático médio);
   - MAE (erro absoluto médio).

# Problema 2 – Regressão Logística para Previsão de Doenças Cardíacas

## Objetivo
Desenvolver um modelo de classificação utilizando regressão logística para prever a presença de doenças cardíacas com base em dados clínicos de pacientes.

## Dataset
Foi utilizado o dataset **Heart Disease Cleveland**, disponibilizado no UCI Machine Learning Repository, contendo 303 registros de pacientes e 13 variáveis clínicas.

**Fonte:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

### Variáveis do dataset

| Variável | Descrição |
|---|---|
| age | Idade do paciente (anos) |
| sex | Sexo biológico (1 = masculino, 0 = feminino) |
| cp | Tipo de dor no peito (1 a 4) |
| trestbps | Pressão arterial em repouso (mmHg) |
| chol | Colesterol sérico total (mg/dl) |
| fbs | Glicemia em jejum acima de 120 mg/dl (1 = sim, 0 = não) |
| restecg | Resultado do eletrocardiograma em repouso (0 a 2) |
| thalach | Frequência cardíaca máxima atingida |
| exang | Angina induzida por exercício (1 = sim, 0 = não) |
| oldpeak | Depressão do segmento ST durante o esforço |
| slope | Inclinação do segmento ST no pico do esforço (1 a 3) |
| ca | Número de artérias coronárias obstruídas (0 a 3) |
| thal | Resultado da cintilografia de tálio (3, 6 ou 7) |

**Variável alvo:**
- `target`: diagnóstico original de 0 a 4, convertido para classificação binária: 0 (sem doença) e 1 (com doença).

## Tarefas realizadas

1. **Pré-processamento:** tratamento de valores ausentes, normalização ou padronização das variáveis independentes, binarização da variável alvo e divisão dos dados em conjuntos de treinamento e teste.

2. **Treinamento do modelo:** implementação de um modelo `LogisticRegression` do Scikit-learn utilizando os seguintes parâmetros:
   - `solver='liblinear'`
   - `random_state=42`
   - `max_iter=1000`

3. **Convergência:** identificação da quantidade de iterações necessárias para a convergência do modelo.

4. **Avaliação do modelo:** cálculo das seguintes métricas utilizando os dados de teste:
   - Acurácia (Accuracy);
   - Precisão (Precision);
   - Revocação (Recall);
   - F1-score;
   - Curva ROC e AUC.
   
# Problema 3 – Classificação de Qualidade de Vinhos Utilizando k-NN

## Objetivo
Desenvolver um modelo de classificação utilizando o algoritmo k-NN (k-Nearest Neighbors) para classificar vinhos tintos em três categorias de qualidade (baixa, média e alta), com base em suas propriedades físico-químicas.

## Dataset
Foi utilizado o arquivo `winequality-red.csv`, pertencente ao dataset **Wine Quality**, disponibilizado no UCI Machine Learning Repository. Contém 1.599 amostras de vinhos tintos da região do Vinho Verde, em Portugal, com 11 atributos físico-químicos.

**Fonte:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

### Variáveis do dataset

| Variável | Descrição |
|---|---|
| fixed acidity | Acidez fixa |
| volatile acidity | Acidez volátil |
| citric acid | Teor de ácido cítrico |
| residual sugar | Açúcar residual |
| chlorides | Teor de cloretos |
| free sulfur dioxide | Dióxido de enxofre livre |
| total sulfur dioxide | Dióxido de enxofre total |
| density | Densidade |
| pH | Nível de acidez (pH) |
| sulphates | Teor de sulfatos |
| alcohol | Teor alcoólico |
| quality | Nota de qualidade atribuída por especialistas (variável alvo) |

**Recodificação da variável alvo (`quality`):**
- Baixa (ruim): notas ≤ 5;
- Média: notas 6–7;
- Alta (boa): notas ≥ 8.

## Tarefas realizadas

1. **Pré-processamento:** carregamento do dataset, recodificação da variável `quality` em três categorias e normalização dos atributos utilizando Min-Max.

2. **Divisão treino/teste:** separação dos dados em 80% para treinamento e 20% para teste, utilizando divisão estratificada para preservar a proporção das classes.

3. **Treinamento e seleção de k:** treinamento de modelos k-NN com diferentes valores de `k` (1, 3, 5, 7 e 11), utilizando validação cruzada de 5 folds. Comparação da acurácia média para selecionar o melhor valor de `k`.

4. **Avaliação do modelo:** utilização do conjunto de teste para calcular:
   - Matriz de confusão;
   - Acurácia (Accuracy);
   - Precisão (Precision) por classe;
   - Revocação (Recall) por classe;
   - F1-score por classe.

5. **Análise dos resultados:** identificação das classes com maior frequência de erros de classificação e análise do comportamento do modelo para valores muito baixos e muito altos de `k`.

# Problema 4 – Classificação de Risco de Crédito Utilizando Árvores de Decisão

## Objetivo
Desenvolver um modelo de classificação utilizando Árvores de Decisão (Decision Tree) para classificar solicitantes de crédito em duas categorias: bom pagador (baixo risco) e mau pagador (alto risco), com base em informações cadastrais e financeiras.

## Dataset
Foi utilizado o dataset **Statlog (German Credit Data)**, disponibilizado no UCI Machine Learning Repository, contendo 1.000 registros de solicitações de crédito, descritos por 20 atributos (7 numéricos e 13 categóricos).

**Fonte:** [UCI German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

### Variáveis do dataset

O dataset possui 20 atributos relacionados ao perfil financeiro e cadastral dos solicitantes, incluindo:

- `checking_status`: situação da conta corrente;
- `duration`: duração do empréstimo (meses);
- `credit_history`: histórico de crédito;
- `purpose`: finalidade do empréstimo;
- `credit_amount`: valor solicitado;
- `savings_status`: situação da poupança;
- `employment`: tempo de emprego atual;
- `installment_commitment`: percentual da renda destinado às parcelas;
- `personal_status`: situação pessoal e sexo;
- `other_parties`: outros responsáveis pelo crédito;
- `residence_since`: tempo de residência atual;
- `property_magnitude`: tipo de propriedade;
- `age`: idade do solicitante;
- `other_payment_plans`: outros planos de pagamento;
- `housing`: condição de moradia;
- `existing_credits`: número de créditos existentes;
- `job`: ocupação profissional;
- `num_dependents`: número de dependentes;
- `own_telephone`: existência de telefone próprio;
- `foreign_worker`: indicação de trabalhador estrangeiro.

**Variável alvo:**
- `class`: classificação do risco de crédito:
  - 1: Bom pagador (baixo risco) – 700 amostras;
  - 2: Mau pagador (alto risco) – 300 amostras.

## Tarefas realizadas

1. **Pré-processamento:** identificação dos atributos numéricos e categóricos, codificação das variáveis categóricas utilizando One-Hot Encoding ou Ordinal Encoding e análise do balanceamento das classes.

2. **Análise de normalização:** avaliação da necessidade de normalização dos atributos, considerando que Árvores de Decisão não dependem da escala das variáveis.

3. **Divisão treino/teste:** separação dos dados em 80% para treinamento e 20% para teste, utilizando divisão estratificada para preservar a proporção das classes.

4. **Treinamento e controle de complexidade:** treinamento de Árvores de Decisão com diferentes valores de profundidade máxima (`max_depth`: 2, 3, 5, 7, 10 e sem limite), utilizando validação cruzada de 5 folds. Comparação das acurácias médias de treinamento e validação para selecionar a melhor profundidade.

5. **Comparação de critérios de divisão:** repetição dos experimentos utilizando os critérios `gini` e `entropy`, avaliando diferenças no desempenho dos modelos.

6. **Interpretação do modelo:** geração da visualização gráfica da árvore escolhida, análise da importância das variáveis e identificação dos três atributos mais relevantes. Extração e interpretação das regras que levam à folha com maior número de amostras classificadas como mau risco.

7. **Avaliação do modelo:** cálculo das seguintes métricas utilizando o conjunto de teste:
   - Matriz de confusão;
   - Acurácia (Accuracy);
   - Precisão (Precision) por classe;
   - Revocação (Recall) por classe;
   - F1-score por classe.

8. **Análise de erros:** identificação dos erros de classificação mais frequentes, avaliação dos impactos de classificar um mau pagador como bom pagador e proposição de ajustes para reduzir esse tipo de erro.
