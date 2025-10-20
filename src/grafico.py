import matplotlib.pyplot as plt


class graficoClass:
    
    @staticmethod
    def histograma(dados):
        plt.hist(dados, bins=30, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Histograma')
        plt.xlabel('Valores')
        plt.ylabel('Frequência')
        plt.show()
        
        
    

    


