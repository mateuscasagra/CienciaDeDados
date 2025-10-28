import numpy as np
import pandas as pd
class calculosClass:
    
    @staticmethod
    def media(dados: pd.Series):
        if isinstance(dados, list):
            serie = pd.Series(dados)
        else:
            serie = dados
        return np.mean(serie)
        
    @staticmethod
    def desvioPadrao(dados: pd.Series):
        if isinstance(dados, list):
            serie = pd.Series(dados)
        else:
            serie = dados
        return serie.std()
    
    @staticmethod
    def variancia(dados: pd.Series):
        if isinstance(dados, list):
            serie = pd.Series(dados)
        else:
            serie = dados
        return serie.var()
    
    @staticmethod
    def percentil(dados: pd.Series, p):
        if isinstance(dados, list):
            serie = pd.Series(dados)
        else:
            serie = dados
        return serie.quantile(p/100)
    
    @staticmethod
    def assimetria(dados: pd.Series):
        if isinstance(dados, list):
            serie = pd.Series(dados)
        else:
            serie = dados
        return serie.skew()
    
    @staticmethod
    def curtose(dados: pd.Series):
        if isinstance(dados, list):
            serie = pd.Series(dados)
        else:
            serie = dados
        return serie.kurtosis()
            
    @staticmethod
    def contagemCondicaoMetereologica(dados):
          if isinstance(dados, pd.DataFrame):
              serie = dados['condicao_metereologica']
          elif isinstance(dados, pd.Series):
              serie = dados
          elif isinstance(dados, list):
              if len(dados) > 0 and isinstance(dados[0], dict):
                  serie = pd.Series([item.get('condicao_metereologica') for item in dados])
              else:
                  serie = pd.Series(dados)
          else:
              serie = pd.Series(dados)
        
          serie = serie.dropna()
          return serie.value_counts().to_dict()