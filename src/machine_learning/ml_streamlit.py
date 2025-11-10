import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .clustering import AnaliseCluster
from .classificacao import AnaliseClassificacao
from .metricas import MetricasAvaliacao
from .visualizacoes_ml import VisualizacoesML

class MLStreamlit:
    """
    Classe para integração das análises de Machine Learning com Streamlit.
    Fornece métodos otimizados para visualização interativa.
    """
    
    def __init__(self):
        self.dados = None
        self.resultados_clustering = None
        self.resultados_classificacao = None
        self.metricas = None
        
    def carregar_dados(self, dados):
        """Carrega os dados para análise."""
        self.dados = pd.DataFrame(dados)
        return len(self.dados)
    
    def executar_clustering(self, n_clusters=3):
        """
        Executa análise de clustering e retorna resultados para visualização.
        """
        try:
            analise = AnaliseCluster()
            
            # Executar clustering
            labels, centroids, dados_processados = analise.executar_kmeans(self.dados, n_clusters)
            
            # Preparar dados para PCA (visualização 2D)
            scaler = StandardScaler()
            dados_scaled = scaler.fit_transform(dados_processados)
            
            pca = PCA(n_components=2)
            dados_pca = pca.fit_transform(dados_scaled)
            
            # Analisar características dos clusters
            caracteristicas = analise.analisar_clusters(self.dados, labels)
            
            self.resultados_clustering = {
                'labels': labels,
                'centroids': centroids,
                'dados_pca': dados_pca,
                'caracteristicas': caracteristicas,
                'variancia_explicada': pca.explained_variance_ratio_
            }
            
            return self.resultados_clustering
            
        except Exception as e:
            st.error(f"Erro no clustering: {e}")
            return None
    
    def executar_classificacao(self):
        """
        Executa análise de classificação e retorna resultados.
        """
        try:
            analise = AnaliseClassificacao()
            
            # Preparar dados
            X, y = analise.preparar_dados(self.dados)
            
            # Treinar modelos
            modelos = analise.treinar_modelos(X, y)
            
            # Avaliar modelos
            resultados = analise.avaliar_modelos(modelos, X, y)
            
            # Obter importância das features (Random Forest)
            if 'random_forest' in modelos:
                feature_names = analise.obter_nomes_features()
                importances = modelos['random_forest'].feature_importances_
                
                # Ordenar por importância
                indices = np.argsort(importances)[::-1]
                
                feature_importance = {
                    'features': [feature_names[i] for i in indices[:10]],  # Top 10
                    'importances': importances[indices[:10]]
                }
            else:
                feature_importance = None
            
            self.resultados_classificacao = {
                'modelos': modelos,
                'resultados': resultados,
                'comparacao': resultados,
                'feature_importance': feature_importance
            }
            
            return self.resultados_classificacao
            
        except Exception as e:
            st.error(f"Erro na classificação: {e}")
            return None
    
    def calcular_metricas(self):
        """
        Calcula métricas detalhadas dos modelos.
        """
        try:
            if not self.resultados_classificacao:
                return None
                
            metricas_calc = MetricasAvaliacao()
            
            # Encontrar melhor modelo
            melhor_modelo = max(
                self.resultados_classificacao['resultados'].items(),
                key=lambda x: x[1]['f1']
            )
            
            self.metricas = {
                'melhor_modelo': {
                    'nome': melhor_modelo[0],
                    'metricas': melhor_modelo[1],
                    'matriz_confusao': melhor_modelo[1].get('matriz_confusao', None)
                },
                'ranking': sorted(
                    self.resultados_classificacao['resultados'].items(),
                    key=lambda x: x[1]['f1'],
                    reverse=True
                )
            }
            
            return self.metricas
            
        except Exception as e:
            st.error(f"Erro no cálculo de métricas: {e}")
            return None
    
    def criar_grafico_clusters(self):
        """
        Cria gráfico interativo dos clusters para Streamlit.
        """
        if not self.resultados_clustering:
            return None
            
        try:
            dados_pca = self.resultados_clustering['dados_pca']
            labels = self.resultados_clustering['labels']
            
            # Criar DataFrame para Plotly
            df_plot = pd.DataFrame({
                'PC1': dados_pca[:, 0],
                'PC2': dados_pca[:, 1],
                'Cluster': [f'Cluster {i}' for i in labels]
            })
            
            fig = px.scatter(
                df_plot, 
                x='PC1', 
                y='PC2', 
                color='Cluster',
                title='Clusters de Acidentes (Análise PCA)',
                labels={'PC1': 'Primeira Componente Principal', 'PC2': 'Segunda Componente Principal'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Erro ao criar gráfico de clusters: {e}")
            return None
    
    def criar_grafico_performance(self):
        """
        Cria gráfico de performance dos modelos.
        """
        if not self.resultados_classificacao:
            return None
            
        try:
            resultados = self.resultados_classificacao['resultados']
            
            modelos = list(resultados.keys())
            metricas_nomes = ['accuracy', 'precision', 'recall', 'f1']
            
            fig = go.Figure()
            
            for metrica in metricas_nomes:
                valores = [resultados[modelo][metrica] for modelo in modelos]
                fig.add_trace(go.Bar(
                    name=metrica.capitalize(),
                    x=modelos,
                    y=valores
                ))
            
            fig.update_layout(
                title='Performance dos Modelos de Machine Learning',
                xaxis_title='Modelos',
                yaxis_title='Score',
                barmode='group',
                template='plotly_white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Erro ao criar gráfico de performance: {e}")
            return None
    
    def criar_grafico_importancia(self):
        """
        Cria gráfico de importância das features.
        """
        if not self.resultados_classificacao or not self.resultados_classificacao['feature_importance']:
            return None
            
        try:
            feature_imp = self.resultados_classificacao['feature_importance']
            
            df_imp = pd.DataFrame({
                'Feature': feature_imp['features'],
                'Importância': feature_imp['importances']
            })
            
            fig = px.bar(
                df_imp,
                x='Importância',
                y='Feature',
                orientation='h',
                title='Importância das Variáveis (Random Forest)',
                labels={'Importância': 'Importância Relativa'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Erro ao criar gráfico de importância: {e}")
            return None
    
    def criar_matriz_confusao(self):
        """
        Cria matriz de confusão interativa.
        """
        if not self.metricas or not self.metricas['melhor_modelo']['matriz_confusao']:
            return None
            
        try:
            matriz = self.metricas['melhor_modelo']['matriz_confusao']
            modelo_nome = self.metricas['melhor_modelo']['nome']
            
            fig = px.imshow(
                matriz,
                text_auto=True,
                aspect="auto",
                title=f'Matriz de Confusão - {modelo_nome.title()}',
                labels=dict(x="Predito", y="Real", color="Quantidade")
            )
            
            fig.update_layout(
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Erro ao criar matriz de confusão: {e}")
            return None
    
    def obter_resumo_executivo(self):
        """
        Retorna resumo executivo dos resultados.
        """
        if not self.metricas:
            return None
            
        try:
            melhor = self.metricas['melhor_modelo']
            total_registros = len(self.dados)
            acidentes_fatais = self.dados['mortos'].sum() if 'mortos' in self.dados.columns else 0
            
            resumo = {
                'total_registros': total_registros,
                'acidentes_fatais': acidentes_fatais,
                'taxa_letalidade': (acidentes_fatais / total_registros * 100) if total_registros > 0 else 0,
                'melhor_modelo': melhor['nome'],
                'f1_score': melhor['metricas']['f1'],
                'acuracia': melhor['metricas']['accuracy'],
                'n_clusters': len(np.unique(self.resultados_clustering['labels'])) if self.resultados_clustering else 0
            }
            
            return resumo
            
        except Exception as e:
            st.error(f"Erro ao gerar resumo: {e}")
            return None
    
    def obter_recomendacoes(self):
        """
        Gera recomendações baseadas nos resultados.
        """
        if not self.metricas or not self.resultados_clustering:
            return []
            
        try:
            recomendacoes = []
            
            # Baseado na performance do modelo
            f1_score = self.metricas['melhor_modelo']['metricas']['f1']
            if f1_score > 0.7:
                recomendacoes.append("✅ Modelo apresenta boa capacidade preditiva (F1 > 70%)")
                recomendacoes.append("🎯 Implementar sistema de alerta baseado no modelo")
            else:
                recomendacoes.append("⚠️ Modelo precisa de melhorias (coletar mais dados)")
            
            # Baseado nos clusters
            n_clusters = len(np.unique(self.resultados_clustering['labels']))
            recomendacoes.append(f"📊 {n_clusters} padrões distintos identificados")
            recomendacoes.append("🚨 Focar recursos nos clusters de maior risco")
            
            # Baseado na importância das features
            if self.resultados_classificacao and self.resultados_classificacao['feature_importance']:
                top_feature = self.resultados_classificacao['feature_importance']['features'][0]
                recomendacoes.append(f"🔍 Variável mais importante: {top_feature}")
            
            return recomendacoes
            
        except Exception as e:
            st.error(f"Erro ao gerar recomendações: {e}")
            return []