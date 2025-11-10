"""
Script para gerar gráficos INDIVIDUAIS de classificação
Para usar nos slides separadamente
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.xlsClass import xlsClass
from src.machine_learning.classificacao import AnaliseClassificacao

print("="*80)
print("GERANDO GRÁFICOS INDIVIDUAIS DE CLASSIFICAÇÃO")
print("="*80)

# Carregar dados
print("\n1. Carregando dados...")
leitor = xlsClass('excel/dados.xlsx')
dados = leitor.aplicaRegras()
print(f"✓ {len(dados)} registros carregados")

# Criar instância e treinar modelos
print("\n2. Treinando modelos...")
analise = AnaliseClassificacao(dados)
X_train, X_test, y_train, y_test = analise.treinar_modelos()
print("✓ Modelos treinados")

# Gerar gráficos individuais
print("\n3. Gerando gráficos individuais...")
analise.gerar_graficos_classificacao_individuais()

print("\n✅ GRÁFICOS INDIVIDUAIS GERADOS!")
print("\nArquivos criados em src/machine_learning/:")
print("   • comparacao_metricas_modelos.png")
print("   • matriz_confusao_melhor_modelo.png")
print("   • ranking_modelos_f1.png")
print("   • importancia_features_rf.png")

# Exibir resultados
print("\n" + "="*80)
print("RESULTADOS DOS MODELOS")
print("="*80)

for nome, resultado in analise.resultados.items():
    print(f"\n{nome.upper()}:")
    print(f"  Acurácia: {resultado['acuracia']:.3f} ({resultado['acuracia']*100:.1f}%)")
    print(f"  Precisão: {resultado['precisao']:.3f} ({resultado['precisao']*100:.1f}%)")
    print(f"  Recall: {resultado['recall']:.3f} ({resultado['recall']*100:.1f}%)")
    print(f"  F1-Score: {resultado['f1_score']:.3f} ({resultado['f1_score']*100:.1f}%)")

melhor = max(analise.resultados.keys(), key=lambda x: analise.resultados[x]['f1_score'])
print(f"\n🏆 MELHOR MODELO: {melhor.upper()}")
print(f"   F1-Score: {analise.resultados[melhor]['f1_score']:.3f}")
