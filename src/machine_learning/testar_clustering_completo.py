"""
Script para testar todos os métodos de clustering implementados
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.xlsClass import xlsClass
from src.machine_learning.clustering import AnaliseCluster

print("="*80)
print("TESTE COMPLETO DE CLUSTERING")
print("="*80)

# Carregar dados
print("\n1. Carregando dados...")
leitor = xlsClass('excel/dados.xlsx')
dados = leitor.aplicaRegras()
print(f"✓ {len(dados)} registros carregados")

# Criar instância da análise
print("\n2. Criando instância de análise...")
analise = AnaliseCluster(dados)
print("✓ Instância criada")

# Executar K-Means
print("\n3. Executando K-Means...")
df_kmeans, modelo_kmeans = analise.executar_kmeans(n_clusters=4)
print(f"✓ K-Means concluído - {len(df_kmeans)} registros processados")

# Executar Hierarchical Clustering
print("\n4. Executando Hierarchical Clustering...")
df_hierarchical, modelo_hierarchical = analise.executar_hierarchical_clustering(n_clusters=4)
print(f"✓ Hierarchical concluído - {len(df_hierarchical)} registros processados")

# Executar Expectation Maximization
print("\n5. Executando Expectation Maximization...")
df_em, modelo_em = analise.executar_expectation_maximization(n_components=4)
print(f"✓ EM concluído - {len(df_em)} registros processados")

# Comparar métodos
print("\n6. Comparando métodos...")
comparacao = analise.comparar_metodos_clustering()
print("✓ Comparação concluída")

# Exibir resultados
print("\n" + "="*80)
print("RESULTADOS FINAIS")
print("="*80)

if comparacao is not None:
    print("\nMÉTRICAS DE QUALIDADE:")
    print(comparacao.to_string(index=False))
    
    print("\n🏆 MELHOR MÉTODO:")
    melhor = comparacao.loc[comparacao['Silhouette'].idxmax()]
    print(f"   {melhor['Método']} (Silhouette = {melhor['Silhouette']:.3f})")

print("\n✅ TESTE COMPLETO FINALIZADO!")
print("\nArquivos gerados:")
print("   • dendrograma_hierarchical.png")
print("   • comparacao_metodos_clustering.png")
