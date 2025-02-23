import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df = pd.read_csv('convneXt_relatorio_previsoes_erradas.csv')

# Contar as ocorrências de cada combinação de Rótulo Verdadeiro e Previsão Errada
contagem = df.groupby(['Rótulo Verdadeiro', 'Previsão Errada']).size().unstack(fill_value=0)

# Plotar o gráfico de barras
contagem.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Previsões Erradas por Rótulo Verdadeiro')
plt.xlabel('Rótulo Verdadeiro')
plt.ylabel('Número de Previsões Erradas')
plt.legend(title='Previsão Errada', bbox_to_anchor=(1.05, 1), loc='upper left')

# Salvar o gráfico em um arquivo JPEG
plt.savefig('convneXt_previsoes_erradas.jpg', format='jpeg', dpi=300, bbox_inches='tight')