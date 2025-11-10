import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class VisualizacoesML:
    def __init__(self):
        self.cores_padrao = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        plt.style.use('seaborn-v0_8')
        
    def plotar_clusters_2d(self, dados, labels_cluster, centroides=None, titulo="Análise de Clusters"):
        """Plota clusters em 2D usando PCA - VERSÃO COMBINADA (4 subplots)"""
        # Aplica PCA para reduzir para 2D
        pca = PCA(n_components=2)
        dados_2d = pca.fit_transform(dados)
        
        plt.figure(figsize=(12, 8))
        
        # Plot principal dos clusters
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(dados_2d[:, 0], dados_2d[:, 1], c=labels_cluster, 
                            cmap='viridis', alpha=0.7, s=50)
        
        # Plot centroides se fornecidos
        if centroides is not None:
            centroides_2d = pca.transform(centroides)
            plt.scatter(centroides_2d[:, 0], centroides_2d[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroides')
            plt.legend()
        
        plt.title(f'{titulo} - Visualização PCA')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)')
        plt.colorbar(scatter, label='Cluster')
        
        # Distribuição dos clusters
        plt.subplot(2, 2, 2)
        unique_labels, counts = np.unique(labels_cluster, return_counts=True)
        plt.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
               autopct='%1.1f%%', startangle=90)
        plt.title('Distribuição dos Clusters')
        
        # Variância explicada pelo PCA
        plt.subplot(2, 2, 3)
        plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_)
        plt.title('Variância Explicada por Componente')
        plt.ylabel('Proporção da Variância')
        
        # Características dos clusters
        plt.subplot(2, 2, 4)
        df_clusters = pd.DataFrame(dados)
        df_clusters['Cluster'] = labels_cluster
        
        # Calcula médias por cluster
        medias_cluster = df_clusters.groupby('Cluster').mean()
        
        # Heatmap das características
        sns.heatmap(medias_cluster.T, annot=True, fmt='.2f', cmap='RdYlBu_r')
        plt.title('Características Médias por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Variáveis')
        
        plt.tight_layout()
        plt.savefig('clusters_analise.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AGORA GERA CADA GRÁFICO INDIVIDUALMENTE
        self.gerar_graficos_clusters_individuais(dados_2d, labels_cluster, centroides, pca, dados)
        
        return pca, dados_2d
    
    def gerar_graficos_clusters_individuais(self, dados_2d, labels_cluster, centroides, pca, dados):
        """Gera cada gráfico de cluster individualmente para uso nos slides"""
        
        # 1. Gráfico principal de clusters (scatter plot)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(dados_2d[:, 0], dados_2d[:, 1], c=labels_cluster, 
                            cmap='viridis', alpha=0.7, s=50)
        if centroides is not None:
            centroides_2d = pca.transform(centroides)
            plt.scatter(centroides_2d[:, 0], centroides_2d[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroides')
            plt.legend()
        plt.title('Visualização de Clusters - PCA 2D', fontsize=14, fontweight='bold')
        plt.xlabel(f'Primeira Componente Principal ({pca.explained_variance_ratio_[0]:.1%} da variância)')
        plt.ylabel(f'Segunda Componente Principal ({pca.explained_variance_ratio_[1]:.1%} da variância)')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('src/machine_learning/clusters_pca_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribuição dos clusters (pizza)
        plt.figure(figsize=(8, 8))
        unique_labels, counts = np.unique(labels_cluster, return_counts=True)
        cores = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        plt.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
               autopct='%1.1f%%', startangle=90, colors=cores)
        plt.title('Distribuição dos Clusters', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('src/machine_learning/distribuicao_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Variância explicada pelo PCA
        plt.figure(figsize=(8, 6))
        plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_, color=['#1f77b4', '#ff7f0e'])
        plt.title('Variância Explicada por Componente Principal', fontsize=14, fontweight='bold')
        plt.ylabel('Proporção da Variância')
        plt.ylim(0, 1)
        for i, v in enumerate(pca.explained_variance_ratio_):
            plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('src/machine_learning/variancia_explicada_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap de características por cluster
        plt.figure(figsize=(10, 8))
        df_clusters = pd.DataFrame(dados)
        df_clusters['Cluster'] = labels_cluster
        medias_cluster = df_clusters.groupby('Cluster').mean()
        sns.heatmap(medias_cluster.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Valor Médio'})
        plt.title('Características Médias por Cluster', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Variáveis', fontsize=12)
        plt.tight_layout()
        plt.savefig('src/machine_learning/heatmap_caracteristicas_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plotar_importancia_features(self, importancias, nomes_features, titulo="Importância das Variáveis"):
        """Plota importância das features de forma detalhada - VERSÃO COMBINADA"""
        # Ordena por importância
        indices = np.argsort(importancias)[::-1]
        
        plt.figure(figsize=(14, 8))
        
        # Gráfico principal de barras
        plt.subplot(2, 2, 1)
        cores = plt.cm.viridis(np.linspace(0, 1, len(importancias)))
        bars = plt.bar(range(len(importancias)), importancias[indices], color=cores)
        plt.title(titulo)
        plt.xlabel('Variáveis')
        plt.ylabel('Importância')
        plt.xticks(range(len(importancias)), [nomes_features[i] for i in indices], rotation=45)
        
        # Adiciona valores nas barras
        for bar, imp in zip(bars, importancias[indices]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{imp:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico de pizza das top 5
        plt.subplot(2, 2, 2)
        top_5_indices = indices[:5]
        top_5_importancias = importancias[top_5_indices]
        top_5_nomes = [nomes_features[i] for i in top_5_indices]
        
        plt.pie(top_5_importancias, labels=top_5_nomes, autopct='%1.1f%%', startangle=90)
        plt.title('Top 5 Variáveis Mais Importantes')
        
        # Gráfico cumulativo
        plt.subplot(2, 2, 3)
        importancias_ordenadas = importancias[indices]
        importancias_cumulativas = np.cumsum(importancias_ordenadas)
        
        plt.plot(range(1, len(importancias) + 1), importancias_cumulativas, 'bo-')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% da importância')
        plt.axhline(y=0.9, color='orange', linestyle='--', label='90% da importância')
        plt.title('Importância Cumulativa')
        plt.xlabel('Número de Variáveis')
        plt.ylabel('Importância Cumulativa')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Tabela de ranking
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        ranking_texto = "RANKING DE IMPORTANCIA:\n\n"
        for i, idx in enumerate(indices[:8], 1):  # Top 8
            ranking_texto += f"{i:2d}o {nomes_features[idx]:<20} {importancias[idx]:.3f}\n"
        
        plt.text(0.1, 0.9, ranking_texto, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AGORA GERA CADA GRÁFICO INDIVIDUALMENTE
        self.gerar_graficos_importancia_individuais(importancias, nomes_features, indices)
        
        return indices
    
    def gerar_graficos_importancia_individuais(self, importancias, nomes_features, indices):
        """Gera cada gráfico de importância individualmente"""
        
        # 1. Gráfico de barras principal (HORIZONTAL para melhor visualização)
        plt.figure(figsize=(10, 8))
        cores = plt.cm.viridis(np.linspace(0, 1, len(importancias)))
        bars = plt.barh(range(len(importancias)), importancias[indices], color=cores)
        plt.yticks(range(len(importancias)), [nomes_features[i] for i in indices])
        plt.xlabel('Importância', fontsize=12, fontweight='bold')
        plt.ylabel('Variáveis', fontsize=12, fontweight='bold')
        plt.title('Importância das Variáveis (Random Forest)', fontsize=14, fontweight='bold')
        
        # Adiciona valores nas barras
        for i, (bar, imp) in enumerate(zip(bars, importancias[indices])):
            plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('src/machine_learning/importancia_features_barras.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gráfico de pizza top 5
        plt.figure(figsize=(10, 8))
        top_5_indices = indices[:5]
        top_5_importancias = importancias[top_5_indices]
        top_5_nomes = [nomes_features[i] for i in top_5_indices]
        cores_pizza = plt.cm.Set3(range(5))
        plt.pie(top_5_importancias, labels=top_5_nomes, autopct='%1.1f%%', 
               startangle=90, colors=cores_pizza, textprops={'fontsize': 12, 'fontweight': 'bold'})
        plt.title('Top 5 Variáveis Mais Importantes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('src/machine_learning/importancia_top5_pizza.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plotar_curvas_aprendizado(self, historico_treino, titulo="Curvas de Aprendizado"):
        """Plota curvas de aprendizado para modelos que têm histórico"""
        if not historico_treino:
            print("Nenhum histórico de treinamento fornecido.")
            return
        
        plt.figure(figsize=(12, 8))
        
        for nome_modelo, historico in historico_treino.items():
            if 'loss' in historico:
                plt.subplot(2, 2, 1)
                plt.plot(historico['loss'], label=f'{nome_modelo} - Loss')
                plt.title('Evolução da Loss')
                plt.xlabel('Época')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            if 'accuracy' in historico:
                plt.subplot(2, 2, 2)
                plt.plot(historico['accuracy'], label=f'{nome_modelo} - Acurácia')
                plt.title('Evolução da Acurácia')
                plt.xlabel('Época')
                plt.ylabel('Acurácia')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('curvas_aprendizado.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plotar_distribuicao_dados(self, dados, titulo="Distribuição dos Dados"):
        """Plota distribuição das variáveis nos dados"""
        df = pd.DataFrame(dados)
        
        # Calcula número de subplots necessários
        n_vars = len(df.columns)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, coluna in enumerate(df.columns, 1):
            plt.subplot(n_rows, n_cols, i)
            
            if df[coluna].dtype in ['object', 'category']:
                # Variável categórica
                contagens = df[coluna].value_counts()
                plt.bar(range(len(contagens)), contagens.values)
                plt.xticks(range(len(contagens)), contagens.index, rotation=45)
                plt.title(f'Distribuição: {coluna}')
                plt.ylabel('Frequência')
            else:
                # Variável numérica
                plt.hist(df[coluna], bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'Distribuição: {coluna}')
                plt.xlabel(coluna)
                plt.ylabel('Frequência')
        
        plt.suptitle(titulo, fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('distribuicao_dados.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plotar_correlacoes(self, dados, titulo="Matriz de Correlação"):
        """Plota matriz de correlação das variáveis - VERSÃO COMBINADA"""
        df = pd.DataFrame(dados)
        
        # Seleciona apenas variáveis numéricas
        df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            print("Nenhuma variavel numerica encontrada para correlacao.")
            return None
        
        plt.figure(figsize=(12, 10))
        
        # Matriz de correlação
        corr_matrix = df_numeric.corr()
        
        # Subplot 1: Heatmap completo
        plt.subplot(2, 2, 1)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, square=True, cbar_kws={'label': 'Correlacao'})
        plt.title('Matriz de Correlacao Completa')
        
        # Subplot 2: Correlações mais fortes
        plt.subplot(2, 2, 2)
        # Pega correlações acima de 0.5 (em valor absoluto)
        mask = np.abs(corr_matrix) > 0.5
        sns.heatmap(corr_matrix, mask=~mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True)
        plt.title('Correlacoes Fortes (|r| > 0.5)')
        
        # Subplot 3: Distribuição das correlações
        plt.subplot(2, 2, 3)
        # Pega apenas o triângulo superior (sem diagonal)
        correlacoes = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        plt.hist(correlacoes, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.title('Distribuicao das Correlacoes')
        plt.xlabel('Correlacao')
        plt.ylabel('Frequencia')
        
        # Subplot 4: Top correlações
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Encontra as correlações mais fortes
        correlacoes_abs = np.abs(corr_matrix.values)
        np.fill_diagonal(correlacoes_abs, 0)  # Remove diagonal
        
        # Pega os índices das correlações mais fortes
        indices_max = np.unravel_index(np.argsort(correlacoes_abs.ravel())[-10:], 
                                      correlacoes_abs.shape)
        
        texto_correlacoes = "TOP 10 CORRELACOES:\n\n"
        for i in range(len(indices_max[0])-1, -1, -1):  # Ordem decrescente
            row, col = indices_max[0][i], indices_max[1][i]
            if row != col:  # Evita diagonal
                var1 = df_numeric.columns[row]
                var2 = df_numeric.columns[col]
                corr_val = corr_matrix.iloc[row, col]
                texto_correlacoes += f"{var1} <-> {var2}: {corr_val:.3f}\n"
        
        plt.text(0.1, 0.9, texto_correlacoes, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(titulo, fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AGORA GERA GRÁFICO INDIVIDUAL PRINCIPAL
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, square=True, cbar_kws={'label': 'Correlacao'}, 
                   linewidths=0.5, linecolor='gray')
        plt.title('Matriz de Correlacao - Variáveis Numéricas', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('src/machine_learning/matriz_correlacao.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix
    
    def plotar_dashboard_ml(self, resultados_clustering, resultados_classificacao, dados_originais):
        """Cria um dashboard completo com todos os resultados de ML"""
        fig = plt.figure(figsize=(20, 16))
        
        # Título principal
        fig.suptitle('DASHBOARD MACHINE LEARNING - ANÁLISE DE ACIDENTES FATAIS', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Clusters (canto superior esquerdo)
        ax1 = plt.subplot(3, 4, 1)
        if 'dados_2d' in resultados_clustering:
            scatter = plt.scatter(resultados_clustering['dados_2d'][:, 0], 
                                resultados_clustering['dados_2d'][:, 1],
                                c=resultados_clustering['labels'], 
                                cmap='viridis', alpha=0.7, s=30)
            plt.title('Clusters de Acidentes', fontweight='bold')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # 2. Distribuição de clusters
        ax2 = plt.subplot(3, 4, 2)
        if 'labels' in resultados_clustering:
            unique_labels, counts = np.unique(resultados_clustering['labels'], return_counts=True)
            plt.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
                   autopct='%1.1f%%', startangle=90)
            plt.title('Distribuição dos Clusters', fontweight='bold')
        
        # 3. Comparação de modelos de classificação
        ax3 = plt.subplot(3, 4, 3)
        if resultados_classificacao:
            modelos = list(resultados_classificacao.keys())
            f1_scores = [resultados_classificacao[m]['f1_score'] for m in modelos]
            
            cores = ['gold', 'silver', '#CD7F32', 'lightcoral'][:len(modelos)]
            bars = plt.bar(range(len(modelos)), f1_scores, color=cores)
            plt.title('F1-Score por Modelo', fontweight='bold')
            plt.ylabel('F1-Score')
            plt.xticks(range(len(modelos)), [m.replace('_', ' ').title() for m in modelos], 
                      rotation=45)
            
            # Adiciona valores nas barras
            for bar, score in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Matriz de confusão do melhor modelo
        ax4 = plt.subplot(3, 4, 4)
        if resultados_classificacao:
            melhor_modelo = max(resultados_classificacao.keys(), 
                              key=lambda x: resultados_classificacao[x]['f1_score'])
            cm = resultados_classificacao[melhor_modelo]['matriz_confusao']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                       xticklabels=['Não Fatal', 'Fatal'],
                       yticklabels=['Não Fatal', 'Fatal'])
            plt.title(f'Matriz de Confusão\n{melhor_modelo.replace("_", " ").title()}', 
                     fontweight='bold')
        
        # 5-8. Distribuições das variáveis principais
        df = pd.DataFrame(dados_originais)
        variaveis_principais = ['fase_dia', 'condicao_metereologica', 'causa_acidente', 'idade']
        
        for i, var in enumerate(variaveis_principais, 5):
            ax = plt.subplot(3, 4, i)
            
            if var in df.columns:
                if df[var].dtype in ['object', 'category'] or var != 'idade':
                    contagens = df[var].value_counts().head(8)  # Top 8
                    plt.bar(range(len(contagens)), contagens.values, alpha=0.7)
                    plt.xticks(range(len(contagens)), contagens.index, rotation=45)
                    plt.title(f'Distribuição: {var.replace("_", " ").title()}', fontweight='bold')
                    plt.ylabel('Frequência')
                else:
                    plt.hist(df[var], bins=15, alpha=0.7, edgecolor='black')
                    plt.title(f'Distribuição: {var.replace("_", " ").title()}', fontweight='bold')
                    plt.xlabel(var.replace("_", " ").title())
                    plt.ylabel('Frequência')
        
        # 9. Importância das variáveis (se disponível)
        ax9 = plt.subplot(3, 4, 9)
        if 'random_forest' in resultados_classificacao and hasattr(resultados_classificacao, 'importancias'):
            importancias = resultados_classificacao['random_forest'].get('importancias', [])
            if importancias:
                features = ['Fase Dia', 'Clima', 'Causa', 'Tipo', 'Idade', 'Sexo']
                plt.barh(features, importancias)
                plt.title('Importância das Variáveis', fontweight='bold')
                plt.xlabel('Importância')
        
        # 10. Estatísticas gerais
        ax10 = plt.subplot(3, 4, 10)
        plt.axis('off')
        
        total_acidentes = len(df)
        acidentes_fatais = df['mortos'].sum() if 'mortos' in df.columns else 0
        taxa_letalidade = (acidentes_fatais / total_acidentes * 100) if total_acidentes > 0 else 0
        
        stats_texto = f"""
        ESTATÍSTICAS GERAIS:
        
        📊 Total de Acidentes: {total_acidentes:,}
        💀 Acidentes Fatais: {acidentes_fatais:,}
        📈 Taxa de Letalidade: {taxa_letalidade:.1f}%
        
        🏆 MELHOR MODELO:
        {melhor_modelo.replace('_', ' ').title() if resultados_classificacao else 'N/A'}
        
        🎯 F1-Score: {resultados_classificacao[melhor_modelo]['f1_score']:.3f if resultados_classificacao else 'N/A'}
        
        📍 CLUSTERS IDENTIFICADOS:
        {len(np.unique(resultados_clustering['labels'])) if 'labels' in resultados_clustering else 'N/A'} padrões distintos
        """
        
        plt.text(0.1, 0.9, stats_texto, transform=ax10.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 11-12. Gráficos adicionais de análise temporal
        ax11 = plt.subplot(3, 4, 11)
        if 'fase_dia' in df.columns:
            fase_mortos = df.groupby('fase_dia')['mortos'].agg(['count', 'sum']).reset_index()
            fase_mortos['taxa_letalidade'] = (fase_mortos['sum'] / fase_mortos['count'] * 100)
            
            plt.bar(range(len(fase_mortos)), fase_mortos['taxa_letalidade'], alpha=0.7)
            plt.xticks(range(len(fase_mortos)), fase_mortos['fase_dia'], rotation=45)
            plt.title('Taxa de Letalidade por Fase do Dia', fontweight='bold')
            plt.ylabel('Taxa de Letalidade (%)')
        
        ax12 = plt.subplot(3, 4, 12)
        if 'condicao_metereologica' in df.columns:
            clima_mortos = df.groupby('condicao_metereologica')['mortos'].agg(['count', 'sum']).reset_index()
            clima_mortos['taxa_letalidade'] = (clima_mortos['sum'] / clima_mortos['count'] * 100)
            
            # Pega apenas os top 6 para visualização
            clima_top = clima_mortos.nlargest(6, 'count')
            plt.bar(range(len(clima_top)), clima_top['taxa_letalidade'], alpha=0.7)
            plt.xticks(range(len(clima_top)), clima_top['condicao_metereologica'], rotation=45)
            plt.title('Taxa de Letalidade por Clima', fontweight='bold')
            plt.ylabel('Taxa de Letalidade (%)')
        
        plt.tight_layout()
        plt.savefig('dashboard_ml_completo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def criar_dashboard_ml(self, resultados_clustering, resultados_classificacao, metricas):
        """
        Cria um dashboard completo integrando clustering e classificação.
        Retorna a figura do Plotly para uso no Streamlit.
        """
        try:
            # Configurar subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Clusters de Acidentes', 'Performance dos Modelos', 
                              'Importância das Features', 'Matriz de Confusão'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "heatmap"}]]
            )
            
            # 1. Clusters (usando PCA para 2D)
            if 'dados_pca' in resultados_clustering and 'labels' in resultados_clustering:
                dados_pca = resultados_clustering['dados_pca']
                labels = resultados_clustering['labels']
                
                for cluster in np.unique(labels):
                    mask = labels == cluster
                    fig.add_trace(
                        go.Scatter(
                            x=dados_pca[mask, 0],
                            y=dados_pca[mask, 1],
                            mode='markers',
                            name=f'Cluster {cluster}',
                            marker=dict(size=8, opacity=0.7)
                        ),
                        row=1, col=1
                    )
            
            # 2. Performance dos modelos
            if 'comparacao' in resultados_classificacao:
                comp = resultados_classificacao['comparacao']
                modelos = list(comp.keys())
                f1_scores = [comp[modelo]['f1'] for modelo in modelos]
                
                fig.add_trace(
                    go.Bar(
                        x=modelos,
                        y=f1_scores,
                        name='F1-Score',
                        marker_color='lightblue'
                    ),
                    row=1, col=2
                )
            
            # 3. Importância das features (Random Forest)
            if 'feature_importance' in resultados_classificacao:
                features = resultados_classificacao['feature_importance']['features']
                importancias = resultados_classificacao['feature_importance']['importances']
                
                fig.add_trace(
                    go.Bar(
                        x=importancias,
                        y=features,
                        orientation='h',
                        name='Importância',
                        marker_color='lightgreen'
                    ),
                    row=2, col=1
                )
            
            # 4. Matriz de confusão do melhor modelo
            if 'melhor_modelo' in metricas and 'matriz_confusao' in metricas['melhor_modelo']:
                matriz = metricas['melhor_modelo']['matriz_confusao']
                
                fig.add_trace(
                    go.Heatmap(
                        z=matriz,
                        colorscale='Blues',
                        showscale=True
                    ),
                    row=2, col=2
                )
            
            # Layout
            fig.update_layout(
                title_text="Dashboard Machine Learning - Acidentes Fatais",
                showlegend=True,
                height=800,
                template='plotly_white'
            )
            
            print("✅ Dashboard ML completo criado!")
            return fig
            
        except Exception as e:
            print(f"❌ Erro ao criar dashboard ML: {e}")
            return None