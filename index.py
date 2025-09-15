#ler arquivo aplicar regras retornar
from src.xlsClass import xlsClass
from src.calculosClass import calculosClass
leitor = xlsClass('excel\dados.xlsx')


colunas = ['idade']
dadosFiltrados = leitor.trazDados(colunas)
stringResult = str(dadosFiltrados)
f = open('relatorio-de-saida.txt')
print(stringResult)

# desvioPadrao = calculosClass.desvioPadrao(dadosFiltrados)
# print(desvioPadrao)