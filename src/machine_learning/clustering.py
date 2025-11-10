import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

class AnaliseCluster:
    def __init__(self, dados):
        self.dados = dados
        self.df = pd.DataFrame(dados)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.resultados_clustering = {}
        
    def preparar_dados(self):
        """Prepara os dados para clustering"""
        # Seleciona as colunas para clustering
        colunas_cluster = ['fase_dia', 'condicao_metereologica', 'tipo_acidente', 'causa_acidente']
        
        # Remove registros com valores nulos
        df_clean = self.df[colunas_cluster].dropna()
        
        # Codifica variáveis categóricas
        df_encoded = df_clean.copy()
        for coluna in colunas_cluster:
            le = LabelEncoder()
            df_encoded[coluna] = le.fit_transform(df_clean[coluna].astype(str))
            self.label_encoders[coluna] = le
            
        return df_encoded
    
    def executar_kmeans(self, n_clusters=4):
        """Executa K-Means clustering"""
        dados_preparados = self.preparar_dados()
        
        # Normaliza dados
        dados_scaled = self.scaler.fit_transform(dados_preparados)
        
        # Aplica K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(dados_scaled)
        
        # Calcula métricas de qualidade
        silhouette = silhouette_score(dados_scaled, clusters)
        davies_bouldin = davies_bouldin_score(dados_scaled, clusters)
        calinski = calinski_harabasz_score(dados_scaled, clusters)
        
        # Adiciona clusters ao dataframe original
        df_resultado = self.df.copy()
        df_resultado = df_resultado.dropna(subset=['fase_dia', 'condicao_metereologica', 'tipo_acidente', 'causa_acidente'])
        df_resultado['cluster_kmeans'] = clusters
        
        # Armazena resultados
        self.resultados_clustering['kmeans'] = {
            'modelo': kmeans,
            'clusters': clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'n_clusters': n_clusters
        }
        
        print(f"K-Means - Silhouette Score: {silhouette:.3f}")
        print(f"K-Means - Davies-Bouldin Index: {davies_bouldin:.3f}")
        print(f"K-Means - Calinski-Harabasz Score: {calinski:.3f}")
        
        return df_resultado, kmeans
    
    def executar_hierarchical_clustering(self, n_clusters=4, linkage_method='ward'):
        """
        Executa Hierarchical Clustering (Agglomerative)
        
        Args:
            n_clusters: Número de clusters desejado
            linkage_method: 'ward', 'complete', 'average', 'single'
        """
        print("\n" + "="*60)
        print("HIERARCHICAL CLUSTERING (AGGLOMERATIVE)")
        print("="*60)
        
        dados_preparados = self.preparar_dados()
        dados_scaled = self.scaler.fit_transform(dados_preparados)
        
        # Aplica Hierarchical Clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        clusters = hierarchical.fit_predict(dados_scaled)
        
        # Calcula métricas de qualidade
        silhouette = silhouette_score(dados_scaled, clusters)
        davies_bouldin = davies_bouldin_score(dados_scaled, clusters)
        calinski = calinski_harabasz_score(dados_scaled, clusters)
        
        # Adiciona clusters ao dataframe
        df_resultado = self.df.copy()
        df_resultado = df_resultado.dropna(subset=['fase_dia', 'condicao_metereologica', 'tipo_acidente', 'causa_acidente'])
        df_resultado['cluster_hierarchical'] = clusters
        
        # Armazena resultados
        self.resultados_clustering['hierarchical'] = {
            'modelo': hierarchical,
            'clusters': clusters,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'n_clusters': n_clusters,
            'linkage': linkage_method
        }
        
        print(f"Método de Linkage: {linkage_method}")
        print(f"Número de Clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        print(f"Calinski-Harabasz Score: {calinski:.3f}")
        
        # Gera dendrograma
        self._plotar_dendrograma(dados_scaled, linkage_method)
        
        return df_resultado, hierarchical
    
    def executar_expectation_maximization(self, n_components=4, covariance_type='full'):
        """
        Executa Expectation Maximization (Gaussian Mixture Model)
        
        Args:
            n_components: Número de componentes (clusters)
            covariance_type: 'full', 'tied', 'diag', 'spherical'
        """
        print("\n" + "="*60)
        print("EXPECTATION MAXIMIZATION (GAUSSIAN MIXTURE)")
        print("="*60)
        
        dados_preparados = self.preparar_dados()
        dados_scaled = self.scaler.fit_transform(dados_preparados)
        
        # Aplica Gaussian Mixture Model (EM)
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42,
            max_iter=200
        )
        gmm.fit(dados_scaled)
        clusters = gmm.predict(dados_scaled)
        probabilidades = gmm.predict_proba(dados_scaled)
        
        # Calcula métricas de qualidade
        silhouette = silhouette_score(dados_scaled, clusters)
        davies_bouldin = davies_bouldin_score(dados_scaled, clusters)
        calinski = calinski_harabasz_score(dados_scaled, clusters)
        
        # Métricas específicas do GMM
        bic = gmm.bic(dados_scaled)  # Bayesian Information Criterion
        aic = gmm.aic(dados_scaled)  # Akaike Information Criterion
        log_likelihood = gmm.score(dados_scaled)
        
        # Adiciona clusters ao dataframe
        df_resultado = self.df.copy()
        df_resultado = df_resultado.dropna(subset=['fase_dia', 'condicao_metereologica', 'tipo_acidente', 'causa_acidente'])
        df_resultado['cluster_em'] = clusters
        
        # Adiciona probabilidades de pertencimento
        for i in range(n_components):
            df_resultado[f'prob_cluster_{i}'] = probabilidades[:, i]
        
        # Armazena resultados
        self.resultados_clustering['em'] = {
            'modelo': gmm,
            'clusters': clusters,
            'probabilidades': probabilidades,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'bic': bic,
            'aic': aic,
            'log_likelihood': log_likelihood,
            'n_components': n_components,
            'covariance_type': covariance_type
        }
        
        print(f"Tipo de Covariância: {covariance_type}")
        print(f"Número de Componentes: {n_components}")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        print(f"Calinski-Harabasz Score: {calinski:.3f}")
        print(f"BIC (Bayesian Information Criterion): {bic:.2f}")
        print(f"AIC (Akaike Information Criterion): {aic:.2f}")
        print(f"Log-Likelihood: {log_likelihood:.2f}")
        
        return df_resultado, gmm
    
    def _plotar_dendrograma(self, dados_scaled, linkage_method='ward'):
        """Plota dendrograma para Hierarchical Clustering"""
        plt.figure(figsize=(12, 6))
        
        # Calcula linkage
        Z = linkage(dados_scaled, method=linkage_method)
        
        # Plota dendrograma
        dendrogram(Z, truncate_mode='lastp', p=30)
        plt.title(f'Dendrograma - Hierarchical Clustering ({linkage_method})')
        plt.xlabel('Índice da Amostra ou (Tamanho do Cluster)')
        plt.ylabel('Distância')
        plt.tight_layout()
        plt.savefig('dendrograma_hierarchical.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Dendrograma salvo: dendrograma_hierarchical.png")
    
    def comparar_metodos_clustering(self):
        """Compara todos os métodos de clustering implementados"""
        print("\n" + "="*80)
        print("COMPARAÇÃO DE MÉTODOS DE CLUSTERING")
        print("="*80)
        
        if not self.resultados_clustering:
            print("Nenhum método de clustering foi executado ainda.")
            return None
        
        # Cria tabela comparativa
        comparacao = []
        for metodo, resultado in self.resultados_clustering.items():
            comparacao.append({
                'Método': metodo.upper(),
                'N Clusters': resultado['n_clusters'] if 'n_clusters' in resultado else resultado['n_components'],
                'Silhouette': resultado['silhouette'],
                'Davies-Bouldin': resultado['davies_bouldin'],
                'Calinski-Harabasz': resultado['calinski_harabasz']
            })
        
        df_comparacao = pd.DataFrame(comparacao)
        
        print("\nMÉTRICAS DE QUALIDADE:")
        print("-" * 80)
        print(df_comparacao.to_string(index=False))
        print("-" * 80)
        
        # Interpretação das métricas
        print("\nINTERPRETAÇÃO:")
        print("• Silhouette Score: Quanto MAIOR, melhor (range: -1 a 1)")
        print("• Davies-Bouldin Index: Quanto MENOR, melhor")
        print("• Calinski-Harabasz Score: Quanto MAIOR, melhor")
        
        # Identifica melhor método
        melhor_silhouette = df_comparacao.loc[df_comparacao['Silhouette'].idxmax()]
        melhor_davies = df_comparacao.loc[df_comparacao['Davies-Bouldin'].idxmin()]
        melhor_calinski = df_comparacao.loc[df_comparacao['Calinski-Harabasz'].idxmax()]
        
        print("\n🏆 MELHORES MÉTODOS POR MÉTRICA:")
        print(f"• Silhouette Score: {melhor_silhouette['Método']} ({melhor_silhouette['Silhouette']:.3f})")
        print(f"• Davies-Bouldin: {melhor_davies['Método']} ({melhor_davies['Davies-Bouldin']:.3f})")
        print(f"• Calinski-Harabasz: {melhor_calinski['Método']} ({melhor_calinski['Calinski-Harabasz']:.3f})")
        
        # Gera gráfico comparativo
        self._plotar_comparacao_metricas(df_comparacao)
        
        return df_comparacao
    
    def _plotar_comparacao_metricas(self, df_comparacao):
        """Plota gráfico comparativo das métricas de clustering"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Comparação de Métodos de Clustering', fontsize=16, fontweight='bold')
        
        metodos = df_comparacao['Método']
        
        # Silhouette Score (maior é melhor)
        axes[0].bar(metodos, df_comparacao['Silhouette'], color='skyblue')
        axes[0].set_title('Silhouette Score\n(Maior é Melhor)')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Davies-Bouldin Index (menor é melhor)
        axes[1].bar(metodos, df_comparacao['Davies-Bouldin'], color='salmon')
        axes[1].set_title('Davies-Bouldin Index\n(Menor é Melhor)')
        axes[1].set_ylabel('Index')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Calinski-Harabasz Score (maior é melhor)
        axes[2].bar(metodos, df_comparacao['Calinski-Harabasz'], color='lightgreen')
        axes[2].set_title('Calinski-Harabasz Score\n(Maior é Melhor)')
        axes[2].set_ylabel('Score')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparacao_metodos_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico comparativo salvo: comparacao_metodos_clustering.png")
    
    def analisar_clusters(self, df_com_clusters):
        """Analisa as características de cada cluster"""
        analise = {}
        
        for cluster_id in df_com_clusters['cluster'].unique():
            cluster_data = df_com_clusters[df_com_clusters['cluster'] == cluster_id]
            
            analise[cluster_id] = {
                'tamanho': len(cluster_data),
                'percentual': len(cluster_data) / len(df_com_clusters) * 100,
                'fase_dia_mais_comum': cluster_data['fase_dia'].mode().iloc[0] if not cluster_data['fase_dia'].mode().empty else 'N/A',
                'clima_mais_comum': cluster_data['condicao_metereologica'].mode().iloc[0] if not cluster_data['condicao_metereologica'].mode().empty else 'N/A',
                'tipo_acidente_mais_comum': cluster_data['tipo_acidente'].mode().iloc[0] if not cluster_data['tipo_acidente'].mode().empty else 'N/A',
                'causa_mais_comum': cluster_data['causa_acidente'].mode().iloc[0] if not cluster_data['causa_acidente'].mode().empty else 'N/A',
                'taxa_mortalidade': cluster_data['mortos'].mean() * 100,
                'idade_media': cluster_data['idade'].mean() if 'idade' in cluster_data.columns else 0
            }
            
        return analise
    
    def gerar_graficos_cluster(self, df_com_clusters):
        """Gera gráficos para visualizar os clusters"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise de Clusters - Padrões de Acidentes Fatais', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Distribuição dos clusters
        cluster_counts = df_com_clusters['cluster'].value_counts().sort_index()
        axes[0, 0].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Distribuição dos Clusters')
        
        # Gráfico 2: Clusters por fase do dia
        cluster_fase = pd.crosstab(df_com_clusters['cluster'], df_com_clusters['fase_dia'])
        cluster_fase.plot(kind='bar', ax=axes[0, 1], stacked=True)
        axes[0, 1].set_title('Clusters por Fase do Dia')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].legend(title='Fase do Dia', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Gráfico 3: Taxa de mortalidade por cluster
        mortalidade_cluster = df_com_clusters.groupby('cluster')['mortos'].mean() * 100
        axes[1, 0].bar(mortalidade_cluster.index, mortalidade_cluster.values, color='red', alpha=0.7)
        axes[1, 0].set_title('Taxa de Mortalidade por Cluster (%)')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Taxa de Mortalidade (%)')
        
        # Gráfico 4: Clusters por condição meteorológica
        cluster_clima = pd.crosstab(df_com_clusters['cluster'], df_com_clusters['condicao_metereologica'])
        cluster_clima.plot(kind='bar', ax=axes[1, 1], stacked=True)
        axes[1, 1].set_title('Clusters por Condição Meteorológica')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].legend(title='Condição Meteorológica', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('clusters_acidentes_fatais.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def gerar_relatorio_cluster(self, analise_clusters):
        """Gera relatório textual dos clusters"""
        relatorio = "=== RELATÓRIO DE ANÁLISE DE CLUSTERS ===\n\n"
        relatorio += "PADRÕES IDENTIFICADOS NOS ACIDENTES FATAIS DE CURITIBA/PR:\n\n"
        
        for cluster_id, dados in analise_clusters.items():
            relatorio += f"CLUSTER {cluster_id}:\n"
            relatorio += f"  • Tamanho: {dados['tamanho']} acidentes ({dados['percentual']:.1f}%)\n"
            relatorio += f"  • Fase do dia predominante: {dados['fase_dia_mais_comum']}\n"
            relatorio += f"  • Condição climática: {dados['clima_mais_comum']}\n"
            relatorio += f"  • Tipo de acidente: {dados['tipo_acidente_mais_comum']}\n"
            relatorio += f"  • Causa principal: {dados['causa_mais_comum']}\n"
            relatorio += f"  • Taxa de mortalidade: {dados['taxa_mortalidade']:.1f}%\n"
            relatorio += f"  • Idade média das vítimas: {dados['idade_media']:.1f} anos\n\n"
        
        # Identifica o cluster mais perigoso
        cluster_mais_perigoso = max(analise_clusters.keys(), 
                                  key=lambda x: analise_clusters[x]['taxa_mortalidade'])
        
        relatorio += f"🚨 CLUSTER MAIS PERIGOSO: Cluster {cluster_mais_perigoso}\n"
        relatorio += f"   Taxa de mortalidade: {analise_clusters[cluster_mais_perigoso]['taxa_mortalidade']:.1f}%\n\n"
        
        relatorio += "RECOMENDAÇÕES PARA POLÍTICAS PÚBLICAS:\n"
        relatorio += f"• Intensificar fiscalização durante: {analise_clusters[cluster_mais_perigoso]['fase_dia_mais_comum']}\n"
        relatorio += f"• Atenção especial em condições: {analise_clusters[cluster_mais_perigoso]['clima_mais_comum']}\n"
        relatorio += f"• Focar na prevenção de: {analise_clusters[cluster_mais_perigoso]['tipo_acidente_mais_comum']}\n"
        
        return relatorio