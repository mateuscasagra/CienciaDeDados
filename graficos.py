import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gerarHistograma(dados, titulo='Histograma'):
    """Gera histograma da distribuição das idades"""
    plt.figure(figsize=(10, 6))
    plt.hist(dados, bins=20, edgecolor='black')
    plt.title(titulo)
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3)
    plt.savefig('histograma_idades.png')
    plt.close()
    
def gerarBoxPlot(dados, titulo='Box Plot'):
    """Gera box plot mostrando quartis e outliers"""
    plt.figure(figsize=(8, 6))
    plt.boxplot(dados)
    plt.title(titulo)
    plt.ylabel('Idade')
    plt.grid(True, alpha=0.3)
    plt.savefig('boxplot_idades.png')
    plt.close()

def gerarGraficosPorCondicao(df):
    """Gera gráficos comparativos por condição meteorológica"""
    # Box plot comparativo
    plt.figure(figsize=(12, 6))
    df.boxplot(column='idade', by='condicao_metereologica')
    plt.title('Distribuição de Idades por Condição Meteorológica')
    plt.suptitle('')
    plt.xlabel('Condição Meteorológica')
    plt.ylabel('Idade')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('boxplot_por_condicao.png')
    plt.close()
    
    # Gráfico de barras comparando estatísticas
    estatisticas = df.groupby('condicao_metereologica')['idade'].agg(['mean', 'median', 'std'])
    estatisticas.plot(kind='bar', figsize=(12, 6))
    plt.title('Comparativo de Estatísticas por Condição Meteorológica')
    plt.xlabel('Condição Meteorológica')
    plt.ylabel('Idade')
    plt.legend(['Média', 'Mediana', 'Desvio Padrão'])
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparativo_estatisticas.png')
    plt.close()

