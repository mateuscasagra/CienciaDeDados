#ler arquivo aplicar regras retornar
from utilitys.xlsClass import xlsClass
from utilitys.calculosClass import calculosClass
leitor = xlsClass('excel\dados.xlsx')

dadosFiltrados = leitor.aplicaRegras()


print(dadosFiltrados)