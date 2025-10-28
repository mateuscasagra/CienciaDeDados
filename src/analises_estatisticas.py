import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, chi2, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class AnaliseEstatistica:
    """
    Classe para implementar análises estatísticas avançadas para o projeto de 
    detecção de padrões temporais em acidentes fatais.
    """
    
    def __init__(self, dados):
        """
        Inicializa a classe com os dados dos acidentes.
        
        Args:
            dados: DataFrame ou lista com os dados dos acidentes
        """
        if isinstance(dados, list):
            self.dados = pd.DataFrame(dados)
        else:
            self.dados = dados
    
    def teorema_central_limite(self, variavel='idade', tamanho_amostra=30, num_amostras=1000):
        """
        Demonstra o Teorema Central do Limite com os dados de acidentes.
        
        Args:
            variavel: Nome da variável a ser analisada
            tamanho_amostra: Tamanho de cada amostra
            num_amostras: Número de amostras a serem geradas
            
        Returns:
            dict: Resultados da análise do TCL
        """
        print("="*60)
        print("TEOREMA CENTRAL DO LIMITE")
        print("="*60)
        
        # Dados originais
        dados_originais = self.dados[variavel].dropna()
        
        # Gerar amostras e calcular médias
        medias_amostrais = []
        for _ in range(num_amostras):
            amostra = np.random.choice(dados_originais, size=tamanho_amostra, replace=True)
            medias_amostrais.append(np.mean(amostra))
        
        medias_amostrais = np.array(medias_amostrais)
        
        # Estatísticas
        media_original = np.mean(dados_originais)
        desvio_original = np.std(dados_originais)
        media_das_medias = np.mean(medias_amostrais)
        desvio_das_medias = np.std(medias_amostrais)
        erro_padrao_teorico = desvio_original / np.sqrt(tamanho_amostra)
        
        # Teste de normalidade das médias amostrais
        _, p_valor_normalidade = stats.shapiro(medias_amostrais)
        
        resultados = {
            'media_original': media_original,
            'desvio_original': desvio_original,
            'media_das_medias': media_das_medias,
            'desvio_das_medias': desvio_das_medias,
            'erro_padrao_teorico': erro_padrao_teorico,
            'p_valor_normalidade': p_valor_normalidade,
            'medias_amostrais': medias_amostrais
        }
        
        print(f"População original - Média: {media_original:.2f}, Desvio: {desvio_original:.2f}")
        print(f"Médias amostrais - Média: {media_das_medias:.2f}, Desvio: {desvio_das_medias:.2f}")
        print(f"Erro padrão teórico: {erro_padrao_teorico:.2f}")
        print(f"Teste de normalidade (p-valor): {p_valor_normalidade:.4f}")
        
        if p_valor_normalidade > 0.05:
            print("✓ As médias amostrais seguem distribuição normal (TCL confirmado)")
        else:
            print("⚠ As médias amostrais não seguem distribuição normal perfeitamente")
        
        return resultados
    
    def analise_correlacao(self):
        """
        Analisa correlações entre variáveis numéricas dos acidentes.
        
        Returns:
            dict: Matriz de correlações e interpretações
        """
        print("\n" + "="*60)
        print("ANÁLISE DE CORRELAÇÃO")
        print("="*60)
        
        # Selecionar apenas variáveis numéricas
        variaveis_numericas = self.dados.select_dtypes(include=[np.number])
        
        if variaveis_numericas.empty:
            print("Nenhuma variável numérica encontrada para análise de correlação.")
            return {}
        
        # Matriz de correlação de Pearson
        matriz_pearson = variaveis_numericas.corr(method='pearson')
        
        # Matriz de correlação de Spearman
        matriz_spearman = variaveis_numericas.corr(method='spearman')
        
        print("Correlações de Pearson (lineares):")
        print(matriz_pearson.round(3))
        
        print("\nCorrelações de Spearman (monotônicas):")
        print(matriz_spearman.round(3))
        
        # Interpretação das correlações
        print("\nInterpretação das correlações:")
        for i in range(len(matriz_pearson.columns)):
            for j in range(i+1, len(matriz_pearson.columns)):
                var1 = matriz_pearson.columns[i]
                var2 = matriz_pearson.columns[j]
                corr_value = matriz_pearson.iloc[i, j]
                
                if abs(corr_value) > 0.7:
                    intensidade = "forte"
                elif abs(corr_value) > 0.3:
                    intensidade = "moderada"
                else:
                    intensidade = "fraca"
                
                direcao = "positiva" if corr_value > 0 else "negativa"
                
                print(f"  {var1} vs {var2}: {corr_value:.3f} (correlação {intensidade} {direcao})")
        
        return {
            'pearson': matriz_pearson,
            'spearman': matriz_spearman
        }
    
    def analise_distribuicao_normal(self, variavel='idade'):
        """
        Analisa se uma variável segue distribuição normal.
        
        Args:
            variavel: Nome da variável a ser analisada
            
        Returns:
            dict: Resultados dos testes de normalidade
        """
        print("\n" + "="*60)
        print("ANÁLISE DE DISTRIBUIÇÃO NORMAL")
        print("="*60)
        
        dados_var = self.dados[variavel].dropna()
        
        # Testes de normalidade
        shapiro_stat, shapiro_p = stats.shapiro(dados_var)
        ks_stat, ks_p = stats.kstest(dados_var, 'norm', args=(dados_var.mean(), dados_var.std()))
        
        # Parâmetros da distribuição normal
        media = np.mean(dados_var)
        desvio = np.std(dados_var)
        
        # Teste de ajuste à normal
        print(f"Variável analisada: {variavel}")
        print(f"Média: {media:.2f}, Desvio padrão: {desvio:.2f}")
        print(f"\nTestes de normalidade:")
        print(f"  Shapiro-Wilk: estatística={shapiro_stat:.4f}, p-valor={shapiro_p:.4f}")
        print(f"  Kolmogorov-Smirnov: estatística={ks_stat:.4f}, p-valor={ks_p:.4f}")
        
        if shapiro_p > 0.05 and ks_p > 0.05:
            print("✓ A variável segue distribuição normal")
        else:
            print("⚠ A variável não segue distribuição normal")
        
        # Cálculo de probabilidades usando distribuição normal
        prob_menor_media = norm.cdf(media, media, desvio)
        prob_maior_media_mais_desvio = 1 - norm.cdf(media + desvio, media, desvio)
        
        print(f"\nProbabilidades (assumindo normalidade):")
        print(f"  P(X < média): {prob_menor_media:.3f}")
        print(f"  P(X > média + 1σ): {prob_maior_media_mais_desvio:.3f}")
        
        return {
            'media': media,
            'desvio': desvio,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'normal': shapiro_p > 0.05 and ks_p > 0.05
        }
    
    def teste_t_student(self, variavel='idade', grupo='condicao_metereologica'):
        """
        Realiza teste t de Student para comparar grupos.
        
        Args:
            variavel: Variável numérica a ser comparada
            grupo: Variável categórica para agrupar
            
        Returns:
            dict: Resultados do teste t
        """
        print("\n" + "="*60)
        print("TESTE T DE STUDENT")
        print("="*60)
        
        if grupo not in self.dados.columns:
            print(f"Variável de grupo '{grupo}' não encontrada.")
            return {}
        
        grupos_unicos = self.dados[grupo].unique()
        grupos_unicos = [g for g in grupos_unicos if pd.notna(g)]
        
        if len(grupos_unicos) < 2:
            print("Necessário pelo menos 2 grupos para o teste t.")
            return {}
        
        resultados = {}
        
        # Teste t para cada par de grupos
        for i in range(len(grupos_unicos)):
            for j in range(i+1, len(grupos_unicos)):
                grupo1_nome = grupos_unicos[i]
                grupo2_nome = grupos_unicos[j]
                
                grupo1_dados = self.dados[self.dados[grupo] == grupo1_nome][variavel].dropna()
                grupo2_dados = self.dados[self.dados[grupo] == grupo2_nome][variavel].dropna()
                
                if len(grupo1_dados) < 2 or len(grupo2_dados) < 2:
                    continue
                
                # Teste t independente
                t_stat, p_valor = stats.ttest_ind(grupo1_dados, grupo2_dados)
                
                # Estatísticas descritivas
                media1 = np.mean(grupo1_dados)
                media2 = np.mean(grupo2_dados)
                desvio1 = np.std(grupo1_dados)
                desvio2 = np.std(grupo2_dados)
                
                print(f"\nComparação: {grupo1_nome} vs {grupo2_nome}")
                print(f"  Grupo 1 - Média: {media1:.2f}, Desvio: {desvio1:.2f}, N: {len(grupo1_dados)}")
                print(f"  Grupo 2 - Média: {media2:.2f}, Desvio: {desvio2:.2f}, N: {len(grupo2_dados)}")
                print(f"  Estatística t: {t_stat:.4f}")
                print(f"  P-valor: {p_valor:.4f}")
                
                if p_valor < 0.05:
                    print("  ✓ Diferença estatisticamente significativa (p < 0.05)")
                else:
                    print("  ⚠ Diferença não significativa (p ≥ 0.05)")
                
                resultados[f"{grupo1_nome}_vs_{grupo2_nome}"] = {
                    't_stat': t_stat,
                    'p_valor': p_valor,
                    'media1': media1,
                    'media2': media2,
                    'significativo': p_valor < 0.05
                }
        
        return resultados
    
    def teste_qui_quadrado(self, var1, var2):
        """
        Realiza teste qui-quadrado de independência entre duas variáveis categóricas.
        
        Args:
            var1: Primeira variável categórica
            var2: Segunda variável categórica
            
        Returns:
            dict: Resultados do teste qui-quadrado
        """
        print("\n" + "="*60)
        print("TESTE QUI-QUADRADO")
        print("="*60)
        
        if var1 not in self.dados.columns or var2 not in self.dados.columns:
            print(f"Uma das variáveis não foi encontrada.")
            return {}
        
        # Criar tabela de contingência
        tabela_contingencia = pd.crosstab(self.dados[var1], self.dados[var2])
        
        print(f"Teste de independência: {var1} vs {var2}")
        print("\nTabela de Contingência:")
        print(tabela_contingencia)
        
        # Teste qui-quadrado
        chi2_stat, p_valor, dof, expected = stats.chi2_contingency(tabela_contingencia)
        
        print(f"\nResultados do teste:")
        print(f"  Estatística χ²: {chi2_stat:.4f}")
        print(f"  Graus de liberdade: {dof}")
        print(f"  P-valor: {p_valor:.4f}")
        print(f"  Valor crítico (α=0.05): {chi2.ppf(0.95, dof):.4f}")
        
        if p_valor < 0.05:
            print("  ✓ As variáveis são dependentes (p < 0.05)")
        else:
            print("  ⚠ As variáveis são independentes (p ≥ 0.05)")
        
        # Frequências esperadas
        print("\nFrequências esperadas:")
        expected_df = pd.DataFrame(expected, 
                                 index=tabela_contingencia.index, 
                                 columns=tabela_contingencia.columns)
        print(expected_df.round(2))
        
        return {
            'chi2_stat': chi2_stat,
            'p_valor': p_valor,
            'dof': dof,
            'tabela_contingencia': tabela_contingencia,
            'frequencias_esperadas': expected_df,
            'dependentes': p_valor < 0.05
        }
    
    def gerar_relatorio_estatistico(self):
        """
        Gera um relatório completo com todas as análises estatísticas.
        
        Returns:
            dict: Todos os resultados das análises
        """
        print("="*80)
        print("RELATÓRIO COMPLETO DE ANÁLISES ESTATÍSTICAS")
        print("PROJETO: DETECÇÃO DE PADRÕES TEMPORAIS EM ACIDENTES FATAIS")
        print("="*80)
        
        resultados = {}
        
        # 1. Teorema Central do Limite
        if 'idade' in self.dados.columns:
            resultados['tcl'] = self.teorema_central_limite('idade')
        
        # 2. Análise de Correlação
        resultados['correlacao'] = self.analise_correlacao()
        
        # 3. Distribuição Normal
        if 'idade' in self.dados.columns:
            resultados['normalidade'] = self.analise_distribuicao_normal('idade')
        
        # 4. Teste t de Student
        if 'idade' in self.dados.columns and 'condicao_metereologica' in self.dados.columns:
            resultados['teste_t'] = self.teste_t_student('idade', 'condicao_metereologica')
        
        # 5. Teste Qui-quadrado
        if 'fase_dia' in self.dados.columns and 'condicao_metereologica' in self.dados.columns:
            resultados['qui_quadrado'] = self.teste_qui_quadrado('fase_dia', 'condicao_metereologica')
        
        print("\n" + "="*80)
        print("RELATÓRIO CONCLUÍDO")
        print("="*80)
        
        return resultados