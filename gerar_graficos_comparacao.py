"""
Script para gerar graficos de comparacao das regressoes
"""
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("GERANDO GRAFICOS DE COMPARACAO")
print("="*80)

# Dados das regressoes
metodos = ['Linear\nSimples', 'Linear\nMultipla', 'Parabolica', 'Exponencial', 'Bayesiana']
r2_scores = [0.1767, 0.3240, 0.0014, 0.2876, 0.3245]
rmse_scores = [4.2099, 3.8148, 4.5805, 3.7662, 3.8135]

# Cores: destaque para o melhor (Bayesiana)
cores_r2 = ['skyblue', 'lightblue', 'lightcoral', 'lightgreen', 'gold']
cores_rmse = ['skyblue', 'lightblue', 'lightcoral', 'lightgreen', 'gold']

# Grafico 1: Comparacao de R2
print("\n1. Gerando grafico de comparacao de R2...")
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(metodos, r2_scores, color=cores_r2, edgecolor='black', linewidth=1.5)

# Adicionar valores nas barras
for i, (bar, valor) in enumerate(zip(bars, r2_scores)):
    ax.text(valor + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{valor:.3f}', 
            va='center', fontsize=11, fontweight='bold')

# Destacar o melhor
bars[-1].set_edgecolor('darkgoldenrod')
bars[-1].set_linewidth(3)

ax.set_xlabel('R² (Coeficiente de Determinacao)', fontsize=12, fontweight='bold')
ax.set_title('Comparacao de R² - Metodos de Regressao', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(r2_scores) * 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Adicionar legenda
ax.text(0.95, 0.05, 'Melhor: Bayesiana (R² = 0.325)', 
        transform=ax.transAxes, fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5),
        ha='right', fontweight='bold')

plt.tight_layout()
plt.savefig('comparacao_r2_regressoes.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK Salvo: comparacao_r2_regressoes.png")

# Grafico 2: Comparacao de RMSE
print("\n2. Gerando grafico de comparacao de RMSE...")
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(metodos, rmse_scores, color=cores_rmse, edgecolor='black', linewidth=1.5)

# Adicionar valores nas barras
for i, (bar, valor) in enumerate(zip(bars, rmse_scores)):
    ax.text(valor + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{valor:.2f}', 
            va='center', fontsize=11, fontweight='bold')

# Destacar o melhor (menor RMSE = Exponencial)
melhor_rmse_idx = rmse_scores.index(min(rmse_scores))
bars[melhor_rmse_idx].set_edgecolor('darkgreen')
bars[melhor_rmse_idx].set_linewidth(3)

ax.set_xlabel('RMSE (Erro Quadratico Medio)', fontsize=12, fontweight='bold')
ax.set_title('Comparacao de RMSE - Metodos de Regressao', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(rmse_scores) * 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Adicionar legenda
ax.text(0.95, 0.05, f'Melhor: {metodos[melhor_rmse_idx].replace(chr(10), " ")} (RMSE = {min(rmse_scores):.2f})', 
        transform=ax.transAxes, fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
        ha='right', fontweight='bold')

plt.tight_layout()
plt.savefig('comparacao_rmse_regressoes.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK Salvo: comparacao_rmse_regressoes.png")

# Grafico 3: Comparacao lado a lado (R2 e RMSE)
print("\n3. Gerando grafico comparativo combinado...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# R2
bars1 = ax1.barh(metodos, r2_scores, color=cores_r2, edgecolor='black', linewidth=1.5)
bars1[-1].set_edgecolor('darkgoldenrod')
bars1[-1].set_linewidth(3)

for bar, valor in zip(bars1, r2_scores):
    ax1.text(valor + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{valor:.3f}', va='center', fontsize=10, fontweight='bold')

ax1.set_xlabel('R² (maior = melhor)', fontsize=11, fontweight='bold')
ax1.set_title('R² - Coeficiente de Determinacao', fontsize=12, fontweight='bold')
ax1.set_xlim(0, max(r2_scores) * 1.15)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# RMSE
bars2 = ax2.barh(metodos, rmse_scores, color=cores_rmse, edgecolor='black', linewidth=1.5)
bars2[melhor_rmse_idx].set_edgecolor('darkgreen')
bars2[melhor_rmse_idx].set_linewidth(3)

for bar, valor in zip(bars2, rmse_scores):
    ax2.text(valor + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{valor:.2f}', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('RMSE (menor = melhor)', fontsize=11, fontweight='bold')
ax2.set_title('RMSE - Erro Quadratico Medio', fontsize=12, fontweight='bold')
ax2.set_xlim(0, max(rmse_scores) * 1.15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.suptitle('Comparacao Completa - Metodos de Regressao', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('comparacao_completa_regressoes.png', dpi=300, bbox_inches='tight')
plt.close()
print("OK Salvo: comparacao_completa_regressoes.png")

# Grafico 4: Regressao nao linear - curvas
print("\n4. Gerando grafico de regressao nao linear (ilustrativo)...")
from src.xlsClass import xlsClass
from src.regressoes import AnaliseRegressao

leitor = xlsClass('excel/dados.xlsx')
dados = leitor.aplicaRegras()
regressao = AnaliseRegressao(dados)
dados_temporais = regressao.preparar_dados_temporais()

# Executar regressao nao linear parabolica
resultado_parabola = regressao.regressao_nao_linear_parabola('tempo_sequencial', 'num_acidentes', dados_temporais)

if resultado_parabola and 'metodos' in resultado_parabola:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Dados originais
    ax.scatter(resultado_parabola['x'], resultado_parabola['y'], 
               alpha=0.3, s=30, color='gray', label='Dados Reais')
    
    # Ordenar x para plotar linhas suaves
    x_sorted_idx = np.argsort(resultado_parabola['x'])
    x_sorted = resultado_parabola['x'][x_sorted_idx]
    
    # Plotar cada metodo
    cores_metodos = ['red', 'blue', 'green']
    nomes_metodos = ['Minimos Quadrados', 'Gauss-Newton', 'Polinomial']
    
    for i, (metodo, dados_metodo) in enumerate(resultado_parabola['metodos'].items()):
        if 'y_pred' in dados_metodo:
            y_pred_sorted = dados_metodo['y_pred'][x_sorted_idx]
            nome = nomes_metodos[i] if i < len(nomes_metodos) else metodo
            ax.plot(x_sorted, y_pred_sorted, 
                   color=cores_metodos[i % len(cores_metodos)], 
                   linewidth=2.5, 
                   label=f'{nome} (R²={dados_metodo["r2"]:.3f})',
                   alpha=0.8)
    
    ax.set_xlabel('Tempo Sequencial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Numero de Acidentes', fontsize=12, fontweight='bold')
    ax.set_title('Regressao Nao Linear - Parabolica (Comparacao de Metodos)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('regressao_nao_linear_parabolica.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK Salvo: regressao_nao_linear_parabolica.png")

print("\n" + "="*80)
print("TODOS OS GRAFICOS FORAM GERADOS COM SUCESSO!")
print("="*80)
print("\nArquivos gerados:")
print("  1. comparacao_r2_regressoes.png")
print("  2. comparacao_rmse_regressoes.png")
print("  3. comparacao_completa_regressoes.png (R2 + RMSE lado a lado)")
print("  4. regressao_nao_linear_parabolica.png")
print("\nRecomendacao para slides:")
print("  - Use: regressao_linear_simples.png (ja existe)")
print("  - Use: comparacao_completa_regressoes.png (mostra tudo)")
print("  - Ou use: regressao_nao_linear_parabolica.png (mostra curvas)")
