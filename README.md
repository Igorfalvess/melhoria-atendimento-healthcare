Melhoria do Atendimento ao Paciente através da Análise de Dados na HealthCare Solutions
**Foco Principal: Predição do Risco de Readmissão Hospitalar**

Este projeto aplica as técnicas de Ciência de Dados e Machine Learning (Random Forest) para resolver um desafio crítico na HealthCare Solutions (HCS): a alta taxa de readmissão hospitalar em 30 dias. O objetivo é construir um modelo preditivo capaz de identificar, proativamente, os pacientes com maior risco de serem readmitidos antes mesmo da alta, permitindo à HCS otimizar recursos e adotar planos de cuidado pós-alta direcionados.

A iniciativa envolve a coleta de dados simulados, tratamento avançado (One-Hot Encoding, imputação) e a aplicação de um algoritmo robusto para transformar dados dispersos em um "score de risco" acionável.


**Etapa 1 – Coleta e Geração de Dados**
O script principal (analise_predicao_readmissao_final.py) foi utilizado para gerar e processar um dataset fictício de 100 registros de pacientes, simulando uma base hospitalar realista.


**Variáveis Chave Utilizadas:**
* Idade, Sexo, Tipo de Admissão
* Dias de Internação e Comorbidades
* Pressão Arterial, Frequência Cardíaca
* Score de Satisfação do Paciente
* Tempo de Espera e Diagnóstico
* Variável Alvo: Readmissão em até 30 dias (Sim/Não).

*Parâmetros da Simulação (Nova Base N=100)*

Parâmetro	Valor
Total de Pacientes	100
Idade	18 a 90 anos
Dias de Internação	1 a 30 dias
Tempo de Espera	0.5 a 6.0 horas (Simulação em horas)
Índice de Comorbidades	0 a 5
<img width="480" height="171" alt="image" src="https://github.com/user-attachments/assets/84f4282d-8be0-4e3c-bf5e-12cb40cec27d" />


**Etapa 2 – Modelagem Preditiva e Insights Acionáveis**
A análise e modelagem foram consolidadas no script analise_predicao_readmissao_final.py.

Algoritmo Aplicado:O modelo principal é o Random Forest Classifier, escolhido por sua alta capacidade preditiva e, crucialmente, pela habilidade de fornecer o Feature Importance Score (Importância das Variáveis).


**Principais Análises e Resultados Gerados:**

  1.  Análise Exploratória (EDA): Identificação da relação direta entre a taxa de readmissão e Idade Elevada, alto índice de Comorbidades e admissões por Emergência/Urgência.
  
  2.  Métricas de Desempenho: Priorização da métrica Recall (Sensibilidade) da classe "1" (Readmissão), garantindo que pacientes de alto risco não sejam classificados erroneamente como saudáveis.

  3.  Importância das Variáveis: O modelo identificou os fatores de risco com maior peso preditivo para a HCS:
    * Fator Crítico Principal: idade
    * Outros Fatores Chave: comorbidades, dias_internacao, pressao_arterial, tempo_espera.

**Saídas de Visualização:**
*  grafico_analise_exploratoria.png (4 gráficos de EDA consolidados)
* grafico_matriz_confusao.png (Validação do desempenho do modelo)
* grafico_importancia_variaveis.png (Visualização do peso dos fatores de risco)


**Tecnologias Utilizadas**

* Python 3
* NumPy – Geração e manipulação de dados numéricos
* Pandas – Tratamento, limpeza e análise de dados
* Scikit-learn – Implementação do Random Forest Classifier e métricas
* Matplotlib & Seaborn – Criação de visualizações e gráficos



**Resultados e Conclusão**
O projeto demonstra a viabilidade de transformar a HCS em um modelo de saúde preventivo e proativo. O modelo Random Forest gera um conhecimento prático para:

* Redução de Custos: Evitando as dispendiosas readmissões (custo estimado em R$ 8.500,00 por caso evitado).
* Otimização de Recursos: Direcionando o telemonitoramento e suporte pós-alta (Agendamento de consulta de enfermagem em 72h) exclusivamente para o grupo de maior risco, definido pelos fatores críticos.
* Melhoria do Cuidado: Oferecendo um atendimento mais seguro, eficiente e humano, focado na continuidade.


===

Autor: Igor Ferreira Alves
Curso: Engenharia da Computação — UNIFECAF (2025)

