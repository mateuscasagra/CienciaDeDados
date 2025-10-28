import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit, least_squares
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class AnaliseRegressao:
    """
    Classe para implementar análises de regressão linear e não linear
    para o projeto de detecção de padrões temporais em acidentes fatais.
    """
    
    def __init__(self, dados):
        """
        Inicializa a classe com os dados dos acidentes.
        
        Args:
            dados: DataFrame com os dados dos acidentes
        """
        if isinstance(dados, list):
            self.dados = pd.DataFrame(dados)
        else:
            self.dados = dados
        
        self.resultados = {}
    
    def preparar_dados_temporais(self):
        """
        Prepara dados temporais para análise de padrões em acidentes.
        
        Returns:
            dict: Dados preparados para regressão
        """
        # Criar variáveis temporais sintéticas baseadas nos dados existentes
        np.random.seed(42)
        n_pontos = len(self.dados)
        
        # Simular dados temporais (horas do dia, dias da semana, meses)
        horas = np.random.randint(0, 24, n_pontos)
        dias_semana = np.random.randint(1, 8, n_pontos)  # 1-7 (segunda a domingo)
        meses = np.random.randint(1, 13, n_pontos)  # 1-12
        
        # Simular número de acidentes baseado em padrões realistas
        # Mais acidentes em horários de pico, fins de semana, etc.
        acidentes = (
            10 + 
            5 * np.sin(2 * np.pi * horas / 24) +  # Padrão diário
            3 * np.sin(2 * np.pi * dias_semana / 7) +  # Padrão semanal
            2 * np.sin(2 * np.pi * meses / 12) +  # Padrão anual
            np.random.normal(0, 2, n_pontos)  # Ruído
        )
        acidentes = np.maximum(0, acidentes)  # Não pode ser negativo
        
        dados_temporais = pd.DataFrame({
            'hora': horas,
            'dia_semana': dias_semana,
            'mes': meses,
            'num_acidentes': acidentes,
            'tempo_sequencial': range(n_pontos)
        })
        
        return dados_temporais
    
    def regressao_linear_simples(self, x_col, y_col, dados=None):
        """
        Implementa regressão linear simples.
        
        Args:
            x_col: Nome da variável independente
            y_col: Nome da variável dependente
            dados: DataFrame opcional (usa self.dados se None)
            
        Returns:
            dict: Resultados da regressão linear simples
        """
        print("="*60)
        print("REGRESSÃO LINEAR SIMPLES")
        print("="*60)
        
        if dados is None:
            dados = self.dados
        
        # Preparar dados
        X = dados[x_col].values.reshape(-1, 1)
        y = dados[y_col].values
        
        # Remover valores NaN
        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 2:
            print("Dados insuficientes para regressão.")
            return {}
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Ajustar modelo
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        # Predições
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"Variável independente: {x_col}")
        print(f"Variável dependente: {y_col}")
        print(f"Coeficiente angular: {modelo.coef_[0]:.4f}")
        print(f"Intercepto: {modelo.intercept_:.4f}")
        print(f"R² (treino): {r2_train:.4f}")
        print(f"R² (teste): {r2_test:.4f}")
        print(f"RMSE (treino): {rmse_train:.4f}")
        print(f"RMSE (teste): {rmse_test:.4f}")
        
        return {
            'modelo': modelo,
            'coeficiente': modelo.coef_[0],
            'intercepto': modelo.intercept_,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
    
    def regressao_linear_multipla(self, x_cols, y_col, dados=None):
        """
        Implementa regressão linear múltipla.
        
        Args:
            x_cols: Lista de nomes das variáveis independentes
            y_col: Nome da variável dependente
            dados: DataFrame opcional
            
        Returns:
            dict: Resultados da regressão linear múltipla
        """
        print("\n" + "="*60)
        print("REGRESSÃO LINEAR MÚLTIPLA")
        print("="*60)
        
        if dados is None:
            dados = self.dados
        
        # Preparar dados
        X = dados[x_cols].values
        y = dados[y_col].values
        
        # Remover linhas com NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < len(x_cols) + 1:
            print("Dados insuficientes para regressão múltipla.")
            return {}
        
        # Padronizar variáveis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Ajustar modelo
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        # Predições
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"Variáveis independentes: {x_cols}")
        print(f"Variável dependente: {y_col}")
        print(f"Coeficientes:")
        for i, col in enumerate(x_cols):
            print(f"  {col}: {modelo.coef_[i]:.4f}")
        print(f"Intercepto: {modelo.intercept_:.4f}")
        print(f"R² (treino): {r2_train:.4f}")
        print(f"R² (teste): {r2_test:.4f}")
        print(f"RMSE (treino): {rmse_train:.4f}")
        print(f"RMSE (teste): {rmse_test:.4f}")
        
        return {
            'modelo': modelo,
            'coeficientes': modelo.coef_,
            'intercepto': modelo.intercept_,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'scaler': scaler,
            'variaveis': x_cols
        }
    
    def regressao_nao_linear_parabola(self, x_col, y_col, dados=None):
        """
        Implementa regressão não linear parabólica (y = ax² + bx + c).
        
        Args:
            x_col: Nome da variável independente
            y_col: Nome da variável dependente
            dados: DataFrame opcional
            
        Returns:
            dict: Resultados da regressão parabólica
        """
        print("\n" + "="*60)
        print("REGRESSÃO NÃO LINEAR - PARABÓLICA")
        print("="*60)
        
        if dados is None:
            dados = self.dados
        
        def funcao_parabola(x, a, b, c):
            return a * x**2 + b * x + c
        
        # Preparar dados
        x = dados[x_col].values
        y = dados[y_col].values
        
        # Remover valores NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) < 3:
            print("Dados insuficientes para regressão parabólica.")
            return {}
        
        resultados_metodos = {}
        
        # Método 1: Mínimos Quadrados Não Linear
        try:
            popt_ls, _ = curve_fit(funcao_parabola, x, y, method='lm')
            y_pred_ls = funcao_parabola(x, *popt_ls)
            r2_ls = r2_score(y, y_pred_ls)
            rmse_ls = np.sqrt(mean_squared_error(y, y_pred_ls))
            
            resultados_metodos['minimos_quadrados'] = {
                'parametros': popt_ls,
                'r2': r2_ls,
                'rmse': rmse_ls,
                'y_pred': y_pred_ls
            }
            
            print(f"Mínimos Quadrados Não Linear:")
            print(f"  a={popt_ls[0]:.4f}, b={popt_ls[1]:.4f}, c={popt_ls[2]:.4f}")
            print(f"  R²: {r2_ls:.4f}, RMSE: {rmse_ls:.4f}")
        except:
            print("Erro no método de mínimos quadrados não linear.")
        
        # Método 2: Gauss-Newton (aproximação usando Levenberg-Marquardt)
        try:
            popt_gn, _ = curve_fit(funcao_parabola, x, y, method='lm')
            y_pred_gn = funcao_parabola(x, *popt_gn)
            r2_gn = r2_score(y, y_pred_gn)
            rmse_gn = np.sqrt(mean_squared_error(y, y_pred_gn))
            
            resultados_metodos['gauss_newton'] = {
                'parametros': popt_gn,
                'r2': r2_gn,
                'rmse': rmse_gn,
                'y_pred': y_pred_gn
            }
            
            print(f"Gauss-Newton (Levenberg-Marquardt):")
            print(f"  a={popt_gn[0]:.4f}, b={popt_gn[1]:.4f}, c={popt_gn[2]:.4f}")
            print(f"  R²: {r2_gn:.4f}, RMSE: {rmse_gn:.4f}")
        except:
            print("Erro no método Gauss-Newton.")
        
        # Método 3: Regressão Polinomial (sklearn)
        try:
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(x.reshape(-1, 1))
            modelo_poly = LinearRegression()
            modelo_poly.fit(X_poly, y)
            y_pred_poly = modelo_poly.predict(X_poly)
            r2_poly = r2_score(y, y_pred_poly)
            rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
            
            resultados_metodos['polinomial'] = {
                'modelo': modelo_poly,
                'r2': r2_poly,
                'rmse': rmse_poly,
                'y_pred': y_pred_poly
            }
            
            print(f"Regressão Polinomial (sklearn):")
            print(f"  R²: {r2_poly:.4f}, RMSE: {rmse_poly:.4f}")
        except:
            print("Erro na regressão polinomial.")
        
        return {
            'funcao': 'parabola',
            'x': x,
            'y': y,
            'metodos': resultados_metodos
        }
    
    def regressao_nao_linear_exponencial(self, x_col, y_col, dados=None):
        """
        Implementa regressão não linear exponencial (y = a * e^(bx) + c).
        
        Args:
            x_col: Nome da variável independente
            y_col: Nome da variável dependente
            dados: DataFrame opcional
            
        Returns:
            dict: Resultados da regressão exponencial
        """
        print("\n" + "="*60)
        print("REGRESSÃO NÃO LINEAR - EXPONENCIAL")
        print("="*60)
        
        if dados is None:
            dados = self.dados
        
        def funcao_exponencial(x, a, b, c):
            return a * np.exp(b * x) + c
        
        # Preparar dados
        x = dados[x_col].values
        y = dados[y_col].values
        
        # Remover valores NaN e garantir y > 0 para exponencial
        mask = ~(np.isnan(x) | np.isnan(y)) & (y > 0)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 3:
            print("Dados insuficientes para regressão exponencial.")
            return {}
        
        resultados_metodos = {}
        
        # Método 1: Curve Fit com diferentes algoritmos
        try:
            # Estimativas iniciais
            p0 = [1, 0.1, min(y)]
            
            popt_exp, _ = curve_fit(funcao_exponencial, x, y, p0=p0, maxfev=5000)
            y_pred_exp = funcao_exponencial(x, *popt_exp)
            r2_exp = r2_score(y, y_pred_exp)
            rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
            
            resultados_metodos['curve_fit'] = {
                'parametros': popt_exp,
                'r2': r2_exp,
                'rmse': rmse_exp,
                'y_pred': y_pred_exp
            }
            
            print(f"Curve Fit Exponencial:")
            print(f"  a={popt_exp[0]:.4f}, b={popt_exp[1]:.4f}, c={popt_exp[2]:.4f}")
            print(f"  R²: {r2_exp:.4f}, RMSE: {rmse_exp:.4f}")
        except Exception as e:
            print(f"Erro no curve fit exponencial: {e}")
        
        # Método 2: Linearização (ln(y-c) = ln(a) + bx)
        try:
            c_est = min(y) - 1  # Estimativa do parâmetro c
            y_linear = np.log(y - c_est)
            
            # Regressão linear nos dados transformados
            modelo_linear = LinearRegression()
            modelo_linear.fit(x.reshape(-1, 1), y_linear)
            
            # Recuperar parâmetros originais
            a_est = np.exp(modelo_linear.intercept_)
            b_est = modelo_linear.coef_[0]
            
            y_pred_linear = funcao_exponencial(x, a_est, b_est, c_est)
            r2_linear = r2_score(y, y_pred_linear)
            rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
            
            resultados_metodos['linearizacao'] = {
                'parametros': [a_est, b_est, c_est],
                'r2': r2_linear,
                'rmse': rmse_linear,
                'y_pred': y_pred_linear
            }
            
            print(f"Linearização:")
            print(f"  a={a_est:.4f}, b={b_est:.4f}, c={c_est:.4f}")
            print(f"  R²: {r2_linear:.4f}, RMSE: {rmse_linear:.4f}")
        except Exception as e:
            print(f"Erro na linearização: {e}")
        
        return {
            'funcao': 'exponencial',
            'x': x,
            'y': y,
            'metodos': resultados_metodos
        }
    
    def regressao_bayesiana(self, x_cols, y_col, dados=None):
        """
        Implementa regressão Bayesiana.
        
        Args:
            x_cols: Lista de variáveis independentes
            y_col: Variável dependente
            dados: DataFrame opcional
            
        Returns:
            dict: Resultados da regressão Bayesiana
        """
        print("\n" + "="*60)
        print("REGRESSÃO BAYESIANA")
        print("="*60)
        
        if dados is None:
            dados = self.dados
        
        # Preparar dados
        X = dados[x_cols].values
        y = dados[y_col].values
        
        # Remover linhas com NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < len(x_cols) + 1:
            print("Dados insuficientes para regressão Bayesiana.")
            return {}
        
        # Padronizar dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # Modelo Bayesiano
        modelo_bayes = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        modelo_bayes.fit(X_train, y_train)
        
        # Predições
        y_pred_train, y_std_train = modelo_bayes.predict(X_train, return_std=True)
        y_pred_test, y_std_test = modelo_bayes.predict(X_test, return_std=True)
        
        # Métricas
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"Variáveis independentes: {x_cols}")
        print(f"Variável dependente: {y_col}")
        print(f"Alpha (precisão do ruído): {modelo_bayes.alpha_:.4f}")
        print(f"Lambda (precisão dos pesos): {modelo_bayes.lambda_:.4f}")
        print(f"R² (treino): {r2_train:.4f}")
        print(f"R² (teste): {r2_test:.4f}")
        print(f"RMSE (treino): {rmse_train:.4f}")
        print(f"RMSE (teste): {rmse_test:.4f}")
        print(f"Incerteza média (teste): {np.mean(y_std_test):.4f}")
        
        return {
            'modelo': modelo_bayes,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'incerteza_media': np.mean(y_std_test),
            'y_pred_test': y_pred_test,
            'y_std_test': y_std_test,
            'scaler': scaler
        }
    
    def comparar_metodos(self):
        """
        Compara todos os métodos de regressão implementados.
        
        Returns:
            dict: Comparação dos resultados
        """
        print("\n" + "="*80)
        print("COMPARAÇÃO DE MÉTODOS DE REGRESSÃO")
        print("="*80)
        
        # Preparar dados temporais para análise
        dados_temporais = self.preparar_dados_temporais()
        
        comparacao = {}
        
        # Regressão Linear Simples
        resultado_linear = self.regressao_linear_simples('hora', 'num_acidentes', dados_temporais)
        if resultado_linear:
            comparacao['linear_simples'] = {
                'r2': resultado_linear['r2_test'],
                'rmse': resultado_linear['rmse_test'],
                'tipo': 'Linear Simples'
            }
        
        # Regressão Linear Múltipla
        resultado_multipla = self.regressao_linear_multipla(['hora', 'dia_semana'], 'num_acidentes', dados_temporais)
        if resultado_multipla:
            comparacao['linear_multipla'] = {
                'r2': resultado_multipla['r2_test'],
                'rmse': resultado_multipla['rmse_test'],
                'tipo': 'Linear Múltipla'
            }
        
        # Regressão Não Linear - Parabólica
        resultado_parabola = self.regressao_nao_linear_parabola('tempo_sequencial', 'num_acidentes', dados_temporais)
        if resultado_parabola and 'minimos_quadrados' in resultado_parabola['metodos']:
            comparacao['parabola'] = {
                'r2': resultado_parabola['metodos']['minimos_quadrados']['r2'],
                'rmse': resultado_parabola['metodos']['minimos_quadrados']['rmse'],
                'tipo': 'Parabólica'
            }
        
        # Regressão Bayesiana
        resultado_bayes = self.regressao_bayesiana(['hora', 'dia_semana'], 'num_acidentes', dados_temporais)
        if resultado_bayes:
            comparacao['bayesiana'] = {
                'r2': resultado_bayes['r2_test'],
                'rmse': resultado_bayes['rmse_test'],
                'tipo': 'Bayesiana',
                'incerteza': resultado_bayes['incerteza_media']
            }
        
        # Exibir comparação
        print("\nResumo da Comparação:")
        print("-" * 60)
        print(f"{'Método':<20} {'R²':<10} {'RMSE':<10} {'Observações'}")
        print("-" * 60)
        
        for metodo, resultado in comparacao.items():
            observacao = ""
            if 'incerteza' in resultado:
                observacao = f"Incerteza: {resultado['incerteza']:.3f}"
            
            print(f"{resultado['tipo']:<20} {resultado['r2']:<10.4f} {resultado['rmse']:<10.4f} {observacao}")
        
        # Encontrar melhor método
        if comparacao:
            melhor_r2 = max(comparacao.items(), key=lambda x: x[1]['r2'])
            melhor_rmse = min(comparacao.items(), key=lambda x: x[1]['rmse'])
            
            print(f"\nMelhor R²: {melhor_r2[1]['tipo']} (R² = {melhor_r2[1]['r2']:.4f})")
            print(f"Menor RMSE: {melhor_rmse[1]['tipo']} (RMSE = {melhor_rmse[1]['rmse']:.4f})")
        
        return comparacao