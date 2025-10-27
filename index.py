#ler arquivo aplicar regras retornar
from src.xlsClass import xlsClass
from src.calculosClass import calculosClass
from src.grafico import graficoClass
leitor = xlsClass('excel\dados.xlsx')


colunas = ['mortos']
dadosFiltrados = leitor.trazDados(colunas)




desvioPadrao = calculosClass.desvioPadrao(dadosFiltrados)
graficoClass.histograma(desvioPadrao)


# print(desvioPadrao)