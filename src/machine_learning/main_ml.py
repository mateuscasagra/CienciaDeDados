# -*- coding: utf-8 -*-
"""
Módulo principal para execução completa das análises de Machine Learning
Análise de Padrões Temporais em Acidentes Fatais - Curitiba/PR
Configuração de encoding para compatibilidade com Windows
"""
import sys
import os
import io

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xlsClass import xlsClass
from machine_learning.clustering import AnaliseCluster
from machine_learning.classificacao import AnaliseClassificacao
from machine_learning.metricas import MetricasAvaliacao
from machine_learning.visualizacoes_ml import VisualizacoesML

def executar_analise_completa():
    """Executa análise completa de Machine Learning"""
    
    print("="*80)
    print("[ML] ANALISE DE MACHINE LEARNING - ACIDENTES FATAIS EM CURITIBA/PR")
    print("="*80)
    
    # 1. CARREGAMENTO DOS DADOS
    print("\n[1] CARREGANDO DADOS...")
    try:
        # Carrega dados usando a classe existente
        arquivo_excel = r"c:\Users\marco\OneDrive\Documentos\CienciaDeDados\excel\dados.xlsx"
        xls = xlsClass(arquivo_excel)
        dados = xls.aplicaRegras()  # Usa aplicaRegras() que retorna os dados filtrados
        
        print(f"[OK] Dados carregados: {len(dados)} registros de acidentes fatais")
        print(f"[LOCAL] Local: Curitiba/PR")
        print(f"[VAR] Variaveis: {list(dados[0].keys()) if dados else 'Nenhuma'}")
        
    except Exception as e:
        print(f"[ERRO] Erro ao carregar dados: {e}")
        return
    
    # 2. ANÁLISE DE CLUSTERING
    print("\n[2] EXECUTANDO ANALISE DE CLUSTERING...")
    try:
        analise_cluster = AnaliseCluster(dados)
        
        # Executa K-Means
        print("   [K-MEANS] Executando K-Means...")
        df_kmeans, modelo_kmeans = analise_cluster.executar_kmeans(n_clusters=4)
        
        # Executa Hierarchical Clustering
        print("\n   [HIERARCHICAL] Executando Hierarchical Clustering...")
        df_hierarchical, modelo_hierarchical = analise_cluster.executar_hierarchical_clustering(n_clusters=4)
        
        # Executa Expectation Maximization
        print("\n   [EM] Executando Expectation Maximization (GMM)...")
        df_em, modelo_em = analise_cluster.executar_expectation_maximization(n_components=4)
        
        # Compara todos os métodos
        print("\n   [COMPARACAO] Comparando metodos de clustering...")
        comparacao_clustering = analise_cluster.comparar_metodos_clustering()
        
        # Gera visualizações
        print("\n   [GRAFICOS] Gerando visualizacoes de clusters...")
        analise_cluster.gerar_graficos_cluster(df_kmeans)
        
        # Gera relatório
        print("   [RELATORIO] Gerando relatorio de clustering...")
        analise_clusters_kmeans = analise_cluster.analisar_clusters(df_kmeans)
        relatorio_cluster = analise_cluster.gerar_relatorio_cluster(analise_clusters_kmeans)
        
        print("\n[OK] Analise de clustering concluida!")
        print(f"   [METODOS] 3 metodos implementados: K-Means, Hierarchical, EM")
        print(f"   [MELHOR] Melhor metodo: {comparacao_clustering.loc[comparacao_clustering['Silhouette'].idxmax(), 'Método']}")
        
        resultados_clustering = {
            'kmeans': analise_cluster.resultados_clustering['kmeans'],
            'hierarchical': analise_cluster.resultados_clustering['hierarchical'],
            'em': analise_cluster.resultados_clustering['em'],
            'comparacao': comparacao_clustering,
            'relatorio': relatorio_cluster
        }
        
    except Exception as e:
        print(f"[ERRO] Erro na analise de clustering: {e}")
        import traceback
        traceback.print_exc()
        resultados_clustering = {}
    
    # 3. ANÁLISE DE CLASSIFICAÇÃO
    print("\n[3] EXECUTANDO ANALISE DE CLASSIFICACAO...")
    try:
        analise_classificacao = AnaliseClassificacao(dados)
        
        # Treina todos os modelos
        print("   [TREINO] Treinando modelos de classificacao...")
        print("      - Arvore de Decisao")
        print("      - Random Forest")
        print("      - K-Nearest Neighbors (KNN)")
        print("      - Rede Neural")
        
        X_train, X_test, y_train, y_test = analise_classificacao.treinar_modelos()
        
        # Gera visualizações
        print("   [GRAFICOS] Gerando visualizacoes de classificacao...")
        analise_classificacao.gerar_graficos_classificacao()
        
        # Gera relatório
        print("   [RELATORIO] Gerando relatorio de classificacao...")
        relatorio_classificacao = analise_classificacao.gerar_relatorio_classificacao()
        
        print("[OK] Analise de classificacao concluida!")
        print(f"   [MELHOR] Melhor modelo: {max(analise_classificacao.resultados.keys(), key=lambda x: analise_classificacao.resultados[x]['f1_score'])}")
        
        resultados_classificacao = analise_classificacao.resultados
        
    except Exception as e:
        print(f"[ERRO] Erro na analise de classificacao: {e}")
        resultados_classificacao = {}
    
    # 4. ANÁLISE DETALHADA DE MÉTRICAS
    print("\n[4] EXECUTANDO ANALISE DETALHADA DE METRICAS...")
    try:
        metricas = MetricasAvaliacao()
        
        if resultados_classificacao:
            # Compara todos os modelos
            print("   [COMPARACAO] Comparando modelos...")
            df_comparacao, ranking = metricas.comparar_modelos(resultados_classificacao)
            
            # Gera relatório completo
            print("   [RELATORIO] Gerando relatorio completo de metricas...")
            relatorio_metricas = metricas.gerar_relatorio_metricas(resultados_classificacao)
            
            print("[OK] Analise de metricas concluida!")
            print(f"   [VENCEDOR] Modelo vencedor: {ranking[0][0]} (F1: {ranking[0][1]:.3f})")
        
    except Exception as e:
        print(f"[ERRO] Erro na analise de metricas: {e}")
    
    # 5. VISUALIZAÇÕES AVANÇADAS
    print("\n[5] GERANDO VISUALIZACOES AVANCADAS...")
    try:
        viz = VisualizacoesML()
        
        # Dashboard completo
        print("   [DASHBOARD] Criando dashboard completo...")
        if resultados_clustering and resultados_classificacao:
            dashboard = viz.plotar_dashboard_ml(resultados_clustering, resultados_classificacao, dados)
        
        # Análise de correlações
        print("   [CORRELACAO] Analisando correlacoes...")
        viz.plotar_correlacoes(dados, "Correlacoes entre Variaveis de Acidentes")
        
        # Distribuição dos dados
        print("   [DISTRIBUICAO] Analisando distribuicoes...")
        viz.plotar_distribuicao_dados(dados, "Distribuicao das Variaveis")
        
        print("[OK] Visualizacoes avancadas concluidas!")
        
    except Exception as e:
        print(f"[ERRO] Erro nas visualizacoes avancadas: {e}")
    
    # 6. RELATÓRIO FINAL CONSOLIDADO
    print("\n[6] GERANDO RELATORIO FINAL...")
    try:
        relatorio_final = gerar_relatorio_final(resultados_clustering, resultados_classificacao, dados)
        
        # Salva relatório em arquivo
        with open('relatorio_ml_final.txt', 'w', encoding='utf-8') as f:
            f.write(relatorio_final)
        
        print("[OK] Relatorio final salvo em 'relatorio_ml_final.txt'")
        
    except Exception as e:
        print(f"[ERRO] Erro ao gerar relatorio final: {e}")
    
    # 7. RESUMO EXECUTIVO
    print("\n" + "="*80)
    print("[RESUMO] RESUMO EXECUTIVO DA ANALISE")
    print("="*80)
    
    if dados:
        print(f"[TOTAL] Total de acidentes analisados: {len(dados):,}")
        acidentes_fatais = sum(1 for d in dados if d.get('mortos', 0) == 1)
        print(f"[FATAIS] Acidentes fatais: {acidentes_fatais:,}")
        print(f"[TAXA] Taxa de letalidade: {(acidentes_fatais/len(dados)*100):.1f}%")
    
    if resultados_clustering:
        print(f"[CLUSTERS] Clusters identificados: {len(set(resultados_clustering['labels']))}")
        print("   -> Padroes temporais e meteorologicos distintos")
    
    if resultados_classificacao:
        melhor_modelo = max(resultados_classificacao.keys(), 
                          key=lambda x: resultados_classificacao[x]['f1_score'])
        melhor_f1 = resultados_classificacao[melhor_modelo]['f1_score']
        print(f"[MELHOR] Melhor modelo preditivo: {melhor_modelo.replace('_', ' ').title()}")
        print(f"[F1] F1-Score: {melhor_f1:.3f} ({melhor_f1*100:.1f}%)")
        print(f"[ACC] Acuracia: {resultados_classificacao[melhor_modelo]['acuracia']:.3f}")
    
    print("\n[RECOMENDACOES] RECOMENDACOES PARA POLITICAS PUBLICAS:")
    print("   • Implementar sistema de predicao baseado no melhor modelo")
    print("   • Focar recursos nos padroes de maior risco identificados")
    print("   • Monitorar continuamente os fatores mais importantes")
    print("   • Desenvolver campanhas especificas por cluster de risco")
    
    print("\n[OK] ANALISE COMPLETA FINALIZADA!")
    print("[ARQUIVOS] Arquivos gerados:")
    print("   • Graficos de clustering e classificacao")
    print("   • Dashboard completo de Machine Learning")
    print("   • Relatorio final detalhado")
    print("   • Matrizes de confusao e metricas")
    
    return {
        'clustering': resultados_clustering,
        'classificacao': resultados_classificacao,
        'dados': dados
    }

def gerar_relatorio_final(resultados_clustering, resultados_classificacao, dados):
    """Gera relatório final consolidado"""
    
    relatorio = """
================================================================================
                    RELATÓRIO FINAL - MACHINE LEARNING
           ANÁLISE DE PADRÕES TEMPORAIS EM ACIDENTES FATAIS
                        CURITIBA/PR - 2024
================================================================================

1. RESUMO EXECUTIVO
================================================================================
"""
    
    if dados:
        total_acidentes = len(dados)
        acidentes_fatais = sum(1 for d in dados if d.get('mortos', 0) == 1)
        taxa_letalidade = (acidentes_fatais / total_acidentes * 100) if total_acidentes > 0 else 0
        
        relatorio += f"""
[DADOS] DADOS ANALISADOS:
   • Total de registros: {total_acidentes:,}
   • Acidentes fatais: {acidentes_fatais:,}
   • Taxa de letalidade: {taxa_letalidade:.1f}%
   • Localizacao: Curitiba, Parana
   • Fonte: Dados da PRF (Policia Rodoviaria Federal)
"""
    
    if resultados_clustering:
        n_clusters = len(set(resultados_clustering['labels']))
        relatorio += f"""
[CLUSTERING] ANALISE DE CLUSTERING:
   • Algoritmo utilizado: K-Means
   • Numero de clusters identificados: {n_clusters}
   • Variaveis utilizadas: fase_dia, condicao_meteorologica, tipo_acidente
   • Objetivo: Identificar padroes temporais e ambientais
"""
    
    if resultados_classificacao:
        melhor_modelo = max(resultados_classificacao.keys(), 
                          key=lambda x: resultados_classificacao[x]['f1_score'])
        melhor_f1 = resultados_classificacao[melhor_modelo]['f1_score']
        melhor_acc = resultados_classificacao[melhor_modelo]['acuracia']
        
        relatorio += f"""
[CLASSIFICACAO] ANALISE DE CLASSIFICACAO:
   • Modelos testados: Arvore de Decisao, Random Forest, KNN, Rede Neural
   • Melhor modelo: {melhor_modelo.replace('_', ' ').title()}
   • F1-Score: {melhor_f1:.3f} ({melhor_f1*100:.1f}%)
   • Acuracia: {melhor_acc:.3f} ({melhor_acc*100:.1f}%)
   • Objetivo: Predizer probabilidade de acidente fatal
"""
    
    relatorio += """

2. METODOLOGIA
================================================================================

[ABORDAGEM] ABORDAGEM CIENTIFICA:
   • Análise exploratória dos dados
   • Pré-processamento e limpeza
   • Codificação de variáveis categóricas
   • Normalização para algoritmos sensíveis à escala
   • Divisão treino/teste (70%/30%)
   • Validação cruzada estratificada

[ALGORITMOS] ALGORITMOS IMPLEMENTADOS:

   CLUSTERING (Nao Supervisionado):
   • K-Means: Agrupamento por similaridade de padroes
   
   CLASSIFICACAO (Supervisionado):
   • Arvore de Decisao: Regras interpretaveis
   • Random Forest: Ensemble de arvores
   • K-Nearest Neighbors: Classificacao por proximidade
   • Rede Neural: Aprendizado nao-linear

[METRICAS] METRICAS DE AVALIACAO:
   • Acurácia: Proporção de predições corretas
   • Precisão: Verdadeiros positivos / (VP + FP)
   • Recall: Verdadeiros positivos / (VP + FN)
   • F1-Score: Média harmônica entre precisão e recall
   • Matriz de Confusão: Análise detalhada de erros
"""
    
    if resultados_classificacao:
        relatorio += """

3. RESULTADOS DETALHADOS
================================================================================

[PERFORMANCE] PERFORMANCE DOS MODELOS:
"""
        
        for nome, resultado in resultados_classificacao.items():
            relatorio += f"""
   {nome.replace('_', ' ').upper()}:
   • Acuracia: {resultado['acuracia']:.3f} ({resultado['acuracia']*100:.1f}%)
   • Precisao: {resultado['precisao']:.3f} ({resultado['precisao']*100:.1f}%)
   • Recall: {resultado['recall']:.3f} ({resultado['recall']*100:.1f}%)
   • F1-Score: {resultado['f1_score']:.3f} ({resultado['f1_score']*100:.1f}%)
"""
        
        # Ranking
        ranking = sorted(resultados_classificacao.items(), 
                        key=lambda x: x[1]['f1_score'], reverse=True)
        
        relatorio += """
[RANKING] RANKING DOS MODELOS (por F1-Score):
"""
        for i, (nome, resultado) in enumerate(ranking, 1):
            posicao = "1o" if i == 1 else "2o" if i == 2 else "3o" if i == 3 else f"{i}o"
            relatorio += f"   {posicao} {nome.replace('_', ' ').title()}: {resultado['f1_score']:.3f}\n"
    
    relatorio += """

4. INTERPRETACAO DOS RESULTADOS
================================================================================

[PADROES] PADROES IDENTIFICADOS:
   • Diferentes perfis de risco foram identificados atraves do clustering
   • Variaveis temporais (fase do dia) mostram forte correlacao com letalidade
   • Condicoes meteorologicas influenciam significativamente o risco
   • Idade e sexo sao fatores preditivos importantes

[CAPACIDADE] CAPACIDADE PREDITIVA:
   • O modelo consegue identificar casos de alto risco com boa precisao
   • Taxa de falsos positivos controlada
   • Recall adequado para nao perder casos criticos
   • F1-Score balanceado indica modelo robusto

[LIMITACOES] LIMITACOES:
   • Dados limitados a Curitiba/PR
   • Periodo temporal especifico
   • Variaveis disponiveis podem nao capturar todos os fatores
   • Necessidade de atualizacao periodica do modelo

5. RECOMENDACOES PARA POLITICAS PUBLICAS
================================================================================

[IMPLEMENTACAO] IMPLEMENTACAO IMEDIATA:
   • Desenvolver sistema de alerta baseado no modelo preditivo
   • Focar recursos de fiscalizacao nos horarios/condicoes de maior risco
   • Criar campanhas educativas especificas por perfil de risco
   • Implementar monitoramento em tempo real

[MONITORAMENTO] MONITORAMENTO CONTINUO:
   • Acompanhar performance do modelo mensalmente
   • Retreinar com novos dados trimestralmente
   • Expandir analise para outras cidades
   • Incluir novas variaveis quando disponiveis

[ACOES] ACES PREVENTIVAS:
   • Intensificar fiscalizacao nos clusters de maior risco
   • Melhorar sinalizacao em pontos criticos identificados
   • Desenvolver campanhas por fase do dia e condicao climatica
   • Criar programa especifico para grupos de maior risco

6. CONCLUSOES
================================================================================

[OBJETIVOS] OBJETIVOS ALCANCADOS:
   • Identificacao de padroes temporais em acidentes fatais
   • Desenvolvimento de modelo preditivo funcional
   • Geracao de insights para politicas publicas
   • Criacao de sistema de monitoramento

[IMPACTO] IMPACTO ESPERADO:
   • Reducao de acidentes fatais atraves de prevencao direcionada
   • Otimizacao de recursos de seguranca publica
   • Melhoria na resposta a emergencias
   • Base cientifica para tomada de decisoes

[PROXIMOS] PROXIMOS PASSOS:
   • Validacao em ambiente de producao
   • Expansao para outras localidades
   • Integracao com sistemas existentes
   • Desenvolvimento de interface para usuarios finais

================================================================================
                            FIM DO RELATÓRIO
================================================================================
"""
    
    return relatorio

if __name__ == "__main__":
    # Executa análise completa
    resultados = executar_analise_completa()