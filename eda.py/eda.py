# EDA final plug-and-play com matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Criar pasta para gráficos, se não existir
os.makedirs("graficos", exist_ok=True)

# Carregar o dataset
df = pd.read_csv('dataset_healthcare_solutions.csv')

# 1️⃣ Informações gerais
print("Informações do Dataset:")
print(df.info())
print("\n5 primeiras linhas:")
print(df.head())

# 2️⃣ Checar valores nulos
print("\nValores nulos por coluna:")
print(df.isnull().sum())

# 3️⃣ Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# 4️⃣ Função para mostrar e salvar gráfico automaticamente
def mostrar_e_salvar(figura, nome_arquivo, tempo=2):
    plt.draw()                                 # Desenha o gráfico
    plt.pause(tempo)                            # Mostra por X segundos
    figura.savefig(f'graficos/{nome_arquivo}.png')  # Salva PNG
    figura.savefig(f'graficos/{nome_arquivo}.pdf')  # Salva PDF
    plt.close()                                 # Fecha a figura

# 5️⃣ Gráficos

# Histograma da idade com média e mediana
fig = plt.figure(figsize=(8,5))
plt.hist(df['idade'], bins=10, color='skyblue', edgecolor='black')
plt.axvline(df['idade'].mean(), color='red', linestyle='dashed', linewidth=1, label=f"Média: {df['idade'].mean():.1f}")
plt.axvline(df['idade'].median(), color='green', linestyle='dashed', linewidth=1, label=f"Mediana: {df['idade'].median():.1f}")
plt.title('Distribuição de Idade dos Pacientes')
plt.xlabel('Idade')
plt.ylabel('Número de Pacientes')
plt.legend()
mostrar_e_salvar(fig, 'hist_idade')

# Boxplot do custo do atendimento
fig = plt.figure(figsize=(8,5))
plt.boxplot(df['custo_atendimento'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Boxplot do Custo do Atendimento')
mostrar_e_salvar(fig, 'box_custo')

# Scatter plot: tempo de internação x custo com linha de tendência
fig = plt.figure(figsize=(8,5))
plt.scatter(df['tempo_internacao_dias'], df['custo_atendimento'], color='orange')
# Linha de tendência simples (reta aproximada)
z = np.polyfit(df['tempo_internacao_dias'], df['custo_atendimento'], 1)
p = np.poly1d(z)
plt.plot(df['tempo_internacao_dias'], p(df['tempo_internacao_dias']), "r--", label='Tendência')
plt.title('Custo do Atendimento x Tempo de Internação')
plt.xlabel('Tempo de Internação (dias)')
plt.ylabel('Custo do Atendimento')
plt.legend()
mostrar_e_salvar(fig, 'scatter_tempo_custo')

# Heatmap de correlação
corr = df.corr()
fig = plt.figure(figsize=(8,5))
plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar(label='Correlação')
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Mapa de Correlação')
plt.tight_layout()
mostrar_e_salvar(fig, 'correlacao')

print("✅ EDA finalizado! Gráficos exibidos e salvos em PNG e PDF na pasta 'graficos'.")
