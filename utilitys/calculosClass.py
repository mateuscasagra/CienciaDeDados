import numpy as np
import pandas as pd
class calculosClass:
    
    @staticmethod
    def media(dados: pd.Series):
         return np.mean(dados)
        
    