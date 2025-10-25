import pandas as pd
import numpy as np
from faker import Faker
import random

# Inicializando gerador de dados
fake = Faker('pt_BR')
np.random.seed(42)

# Quantidade de registros simulados
n = 100

# Gerando dados simulados
dados = {
    'id_paciente': range(1, n + 1),
    'idade': np.random.randint(18, 90, size=n),
    'sexo': [random.choice(['Masculino', 'Feminino']) for _ in range(n)],
    'tempo_espera_min': np.random.randint(5, 180, size=n),
    'satisfacao_paciente': np.random.randint(1, 6, size=n),
    'diagnostico': [random.choice(['Gripe', 'Covid-19', 'Hipertensão', 'Diabetes', 'Infecção']) for _ in range(n)],
    'dispositivo_monitoramento': [random.choice(['Smartwatch', 'Oxímetro', 'Glicômetro', 'Sem dispositivo']) for _ in range(n)],
    'readmissao_30dias': [random.choice(['Sim', 'Não']) for _ in range(n)],
    'tempo_internacao_dias': np.random.randint(1, 15, size=n),
    'custo_atendimento': np.round(np.random.uniform(500, 15000, size=n), 2)
}

# Criando DataFrame
df = pd.DataFrame(dados)

# Exibindo as primeiras linhas
print(df.head())

# Salvando em CSV
df.to_csv("dataset_healthcare_solutions.csv", index=False, encoding='utf-8-sig')
print("\nArquivo 'dataset_healthcare_solutions.csv' criado com sucesso!")
