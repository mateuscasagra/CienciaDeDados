#ler arquivo aplicar regras retornar
from src.xlsClass import xlsClass
from src.calculosClass import calculosClass
from src.grafico import graficoClass
leitor = xlsClass('excel\dados.xlsx')


colunas = ['idade']
dadosFiltrados = leitor.trazDados(colunas)


media = calculosClass.media(dadosFiltrados)
graficoClass.histograma(media)


# desvioPadrao = calculosClass.desvioPadrao(dadosFiltrados)
# print(desvioPadrao)