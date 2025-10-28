from src.xlsClass import xlsClass
from src.calculosClass import calculosClass
import graficos
leitor = xlsClass('excel/dados.xlsx')


colunas = ['mortos']
dadosFiltrados = leitor.trazDados(colunas)

media = calculosClass.media(dadosFiltrados)
stringResult = f"Média das idades: {media}"
with open('relatorio-de-saida.txt', 'w', encoding='utf-8') as f:
    f.write(stringResult)
    

print("="*60)
print("ESTATÍSTICAS DESCRITIVAS - ACIDENTES FATÁIS CURITIBA/PR")
print("="*60)

desvioPadrao = calculosClass.desvioPadrao(dadosFiltrados)
print(f"\nMédia das idades: {media:.2f} anos")
print(f"Desvio padrão: {desvioPadrao:.2f} anos")

variancia = calculosClass.variancia(dadosFiltrados)
print(f"Variância: {variancia:.2f}")

percentil25 = calculosClass.percentil(dadosFiltrados, 25)
percentil50 = calculosClass.percentil(dadosFiltrados, 50)
percentil75 = calculosClass.percentil(dadosFiltrados, 75)
print(f"\nPercentis:")
print(f"  Q1 (25%): {percentil25:.2f} anos")
print(f"  Q2 (50%/Mediana): {percentil50:.2f} anos")
print(f"  Q3 (75%): {percentil75:.2f} anos")

assimetria = calculosClass.assimetria(dadosFiltrados)
print(f"\nAssimetria: {assimetria:.2f}")

curtose = calculosClass.curtose(dadosFiltrados)
print(f"Curtose: {curtose:.2f}")

dadosCompletos = leitor.aplicaRegras()
contagemMetereologica = calculosClass.contagemCondicaoMetereologica(dadosCompletos)
print("Contagem meteorológica:", contagemMetereologica)

import pandas as pd
df_completo = pd.DataFrame(dadosCompletos)
print("\nMédia de idade por condição meteorológica:")
mediaPorCondicao = df_completo.groupby('condicao_metereologica')['idade'].mean()
for condicao, media in mediaPorCondicao.items():
    print(f"  {condicao}: {media:.2f} anos")

print("\nMediana de idade por condição meteorológica:")
medianaPorCondicao = df_completo.groupby('condicao_metereologica')['idade'].median()
for condicao, mediana in medianaPorCondicao.items():
    print(f"  {condicao}: {mediana:.2f} anos")

print("\nDesvio padrão de idade por condição meteorológica:")
desvioPorCondicao = df_completo.groupby('condicao_metereologica')['idade'].std()
for condicao, desvio in desvioPorCondicao.items():
    print(f"  {condicao}: {desvio:.2f} anos")

print("\nModa de idade por condição meteorológica:")
modaPorCondicao = df_completo.groupby('condicao_metereologica')['idade'].apply(lambda x: x.mode())
for condicao in df_completo['condicao_metereologica'].unique():
    modas = df_completo[df_completo['condicao_metereologica'] == condicao]['idade'].mode()
    if len(modas) > 0:
        print(f"  {condicao}: {', '.join([f'{m:.0f} anos' for m in modas])}")
    else:
        print(f"  {condicao}: sem moda")

print("\nVariância de idade por condição meteorológica:")
varianciaPorCondicao = df_completo.groupby('condicao_metereologica')['idade'].var()
for condicao, variancia in varianciaPorCondicao.items():
    print(f"  {condicao}: {variancia:.2f}")

print("\n" + "="*60)
print("GERANDO GRÁFICOS...")
print("="*60)

import numpy as np
idades_array = np.array([x for x in dadosFiltrados if not pd.isna(x)])

graficos.gerarHistograma(idades_array, 'Histograma de Idades - Acidentes Fatais')
print("Histograma gerado: histograma_idades.png")

graficos.gerarBoxPlot(idades_array, 'Box Plot de Idades - Acidentes Fatais')
print("Box Plot gerado: boxplot_idades.png")

graficos.gerarGraficosPorCondicao(df_completo)
print("Graficos por condicao gerados")

print("\n" + "="*60)
print("ANÁLISE DO COMPORTAMENTO DOS RESULTADOS")
print("="*60)
print(f"\nA idade média das vítimas fatais é {media:.1f} anos.")
print(f"Assimetria de {assimetria:.2f} indica ", end="")
if assimetria > 0:
    print("mais vítimas jovens.")
else:
    print("distribuição equilibrada.")
print(f"Desvio padrão de {desvioPadrao:.1f} anos indica ", end="")
if desvioPadrao > 15:
    print("alta variabilidade de idades.")
else:
    print("idades relativamente concentradas.")
