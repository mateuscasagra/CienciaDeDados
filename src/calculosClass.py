import numpy as np
import pandas as pd
class calculosClass:
    
    @staticmethod
    def media(dados: pd.Series):
         return np.mean(dados)
        
    @staticmethod
    def desvioPadrao(dados: pd.Series):
         for i in dados:
              resultadoIdade += dados['idade']
              ResultadoDivisao = resultadoIdade / len(i)
              DiminuiIdade = ResultadoDivisao - dados['idade']
              total = np.sqrt(DiminuiIdade / len(i))
              return total
            