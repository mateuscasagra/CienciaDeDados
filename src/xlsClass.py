import pandas as pd

class xlsClass:
    colunas = [
        'causa_acidente',
        'uf',
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
        mascara2 = self.df['uf'] == 'PR'
        mascara3 = self.df['municipio'] == 'CURITIBA'
        df_filtrado = self.df[mascara & mascara2 & mascara3]
        retorno = df_filtrado.to_dict('records')
        return retorno
        

    def trazDados(self, colunas):
        todosDados = []
        dados = self.aplicaRegras()
        for i in colunas:
            for x in dados:
                todosDados.append(x[i])

        return todosDados
    

        




    
    

 
