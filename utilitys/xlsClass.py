import pandas as pd

class xlsClass:
    colunas = [
        'causa_acidente',
        'municipio',
        'tipo_acidente',
        'classificacao_acidente',
        'mortos',
        'sexo',
        'idade',
        'fase_dia',
        'condicao_metereologica'
    ]
    def __init__(self, caminho):
        self.df = pd.read_excel(caminho, usecols=self.colunas)
        
        
        
    def aplicaRegras(self):
        retorno = []
        mascara = self.df['classificacao_acidente'] == 'Com Vítimas Fatais'
        df_filtrado = self.df[mascara]
        retorno = df_filtrado.to_dict('records')
        return retorno
        

    
    

 
