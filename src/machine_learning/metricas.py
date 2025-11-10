import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

class MetricasAvaliacao:
    def __init__(self):
        self.resultados = {}
        
    def calcular_metricas_basicas(self, y_true, y_pred, nome_modelo):
        """Calcula métricas básicas de classificação"""
        metricas = {
            'acuracia': accuracy_score(y_true, y_pred),
            'precisao': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'matriz_confusao': confusion_matrix(y_true, y_pred)
        }
        
        self.resultados[nome_modelo] = metricas
        return metricas
    
    def calcular_metricas_avancadas(self, y_true, y_pred_proba, nome_modelo):
        """Calcula métricas avançadas usando probabilidades"""
        if nome_modelo not in self.resultados:
            self.resultados[nome_modelo] = {}
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        self.resultados[nome_modelo].update({
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall
        })
        
        return roc_auc, pr_auc
    
    def gerar_matriz_confusao_detalhada(self, y_true, y_pred, nome_modelo, labels=['Não Fatal', 'Fatal']):
        """Gera matriz de confusão detalhada com interpretação"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calcula métricas derivadas da matriz
        tn, fp, fn, tp = cm.ravel()
        
        detalhes = {
            'verdadeiros_negativos': tn,
            'falsos_positivos': fp,
            'falsos_negativos': fn,
            'verdadeiros_positivos': tp,
            'total_casos': len(y_true),
            'casos_positivos': sum(y_true),
            'casos_negativos': len(y_true) - sum(y_true),
            'taxa_verdadeiros_positivos': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'taxa_falsos_positivos': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'especificidade': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensibilidade': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        return cm, detalhes
    
    def plotar_matriz_confusao(self, y_true, y_pred, nome_modelo, labels=['Não Fatal', 'Fatal']):
        """Plota matriz de confusão com detalhes"""
        cm, detalhes = self.gerar_matriz_confusao_detalhada(y_true, y_pred, nome_modelo, labels)
        
        plt.figure(figsize=(10, 8))
        
        # Subplot 1: Matriz de confusão
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Matriz de Confusão - {nome_modelo}')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        
        # Subplot 2: Métricas em barras
        plt.subplot(2, 2, 2)
        metricas_nomes = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'Especificidade']
        metricas_valores = [
            self.resultados[nome_modelo]['acuracia'],
            self.resultados[nome_modelo]['precisao'],
            self.resultados[nome_modelo]['recall'],
            self.resultados[nome_modelo]['f1_score'],
            detalhes['especificidade']
        ]
        
        cores = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
        bars = plt.bar(metricas_nomes, metricas_valores, color=cores)
        plt.title('Métricas de Performance')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Adiciona valores nas barras
        for bar, valor in zip(bars, metricas_valores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{valor:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        # Subplot 3: Distribuição de casos
        plt.subplot(2, 2, 3)
        casos_reais = ['Não Fatal', 'Fatal']
        casos_valores = [detalhes['casos_negativos'], detalhes['casos_positivos']]
        plt.pie(casos_valores, labels=casos_reais, autopct='%1.1f%%', startangle=90)
        plt.title('Distribuição Real dos Casos')
        
        # Subplot 4: Detalhes da matriz
        plt.subplot(2, 2, 4)
        plt.axis('off')
        texto_detalhes = f"""
        DETALHES DA CLASSIFICAÇÃO:
        
        Total de Casos: {detalhes['total_casos']}
        Casos Fatais: {detalhes['casos_positivos']} ({detalhes['casos_positivos']/detalhes['total_casos']*100:.1f}%)
        Casos Não Fatais: {detalhes['casos_negativos']} ({detalhes['casos_negativos']/detalhes['total_casos']*100:.1f}%)
        
        MATRIZ DE CONFUSÃO:
        Verdadeiros Positivos: {detalhes['verdadeiros_positivos']}
        Verdadeiros Negativos: {detalhes['verdadeiros_negativos']}
        Falsos Positivos: {detalhes['falsos_positivos']}
        Falsos Negativos: {detalhes['falsos_negativos']}
        
        TAXAS:
        Sensibilidade (Recall): {detalhes['sensibilidade']:.3f}
        Especificidade: {detalhes['especificidade']:.3f}
        Taxa de Falsos Positivos: {detalhes['taxa_falsos_positivos']:.3f}
        """
        
        plt.text(0.1, 0.9, texto_detalhes, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'matriz_confusao_{nome_modelo.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm, detalhes
    
    def comparar_modelos(self, modelos_resultados):
        """Compara múltiplos modelos e gera ranking"""
        df_comparacao = pd.DataFrame()
        
        for nome, resultados in modelos_resultados.items():
            df_comparacao[nome] = [
                resultados['acuracia'],
                resultados['precisao'],
                resultados['recall'],
                resultados['f1_score']
            ]
        
        df_comparacao.index = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        
        # Plota comparação
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Heatmap de comparação
        plt.subplot(2, 2, 1)
        sns.heatmap(df_comparacao, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'})
        plt.title('Comparação de Métricas entre Modelos')
        
        # Subplot 2: Ranking por F1-Score
        plt.subplot(2, 2, 2)
        f1_scores = {nome: res['f1_score'] for nome, res in modelos_resultados.items()}
        f1_sorted = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        
        nomes = [nome.replace('_', ' ').title() for nome, _ in f1_sorted]
        scores = [score for _, score in f1_sorted]
        cores = ['gold', 'silver', '#CD7F32', 'lightcoral'][:len(nomes)]
        
        bars = plt.barh(nomes, scores, color=cores)
        plt.title('Ranking por F1-Score')
        plt.xlabel('F1-Score')
        
        for bar, score in zip(bars, scores):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center')
        
        # Subplot 3: Radar chart
        plt.subplot(2, 2, 3, projection='polar')
        
        # Pega o melhor modelo para o radar
        melhor_modelo = f1_sorted[0][0]
        melhor_resultados = modelos_resultados[melhor_modelo]
        
        categorias = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        valores = [
            melhor_resultados['acuracia'],
            melhor_resultados['precisao'],
            melhor_resultados['recall'],
            melhor_resultados['f1_score']
        ]
        
        angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
        valores += valores[:1]  # Fecha o círculo
        angulos += angulos[:1]
        
        plt.plot(angulos, valores, 'o-', linewidth=2, label=melhor_modelo.replace('_', ' ').title())
        plt.fill(angulos, valores, alpha=0.25)
        plt.xticks(angulos[:-1], categorias)
        plt.ylim(0, 1)
        plt.title(f'Perfil do Melhor Modelo\n({melhor_modelo.replace("_", " ").title()})')
        
        # Subplot 4: Resumo estatístico
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Calcula estatísticas
        media_f1 = np.mean([res['f1_score'] for res in modelos_resultados.values()])
        std_f1 = np.std([res['f1_score'] for res in modelos_resultados.values()])
        melhor_f1 = max([res['f1_score'] for res in modelos_resultados.values()])
        pior_f1 = min([res['f1_score'] for res in modelos_resultados.values()])
        
        resumo_texto = f"""
        RESUMO ESTATÍSTICO (F1-Score):
        
        🏆 Melhor Modelo: {f1_sorted[0][0].replace('_', ' ').title()}
        📊 Melhor F1-Score: {melhor_f1:.3f}
        📉 Pior F1-Score: {pior_f1:.3f}
        📈 Média F1-Score: {media_f1:.3f}
        📏 Desvio Padrão: {std_f1:.3f}
        
        RECOMENDAÇÃO:
        Usar {f1_sorted[0][0].replace('_', ' ')} para produção
        
        DIFERENÇA:
        {((melhor_f1 - pior_f1) * 100):.1f}% entre melhor e pior modelo
        """
        
        plt.text(0.1, 0.9, resumo_texto, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AGORA GERA CADA GRÁFICO INDIVIDUALMENTE
        self.gerar_graficos_metricas_individuais(modelos_resultados, df_comparacao, f1_sorted)
        
        return df_comparacao, f1_sorted
    
    def gerar_graficos_metricas_individuais(self, modelos_resultados, df_comparacao, f1_sorted):
        """Gera cada gráfico de métricas individualmente para uso nos slides"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # 1. Heatmap de comparação de métricas
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_comparacao, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray')
        plt.title('Comparação de Métricas entre Modelos', fontsize=14, fontweight='bold')
        plt.xlabel('Modelos', fontsize=12, fontweight='bold')
        plt.ylabel('Métricas', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('src/machine_learning/heatmap_comparacao_metricas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Ranking por F1-Score
        plt.figure(figsize=(10, 8))
        nomes = [nome.replace('_', ' ').title() for nome, _ in f1_sorted]
        scores = [score for _, score in f1_sorted]
        cores = ['gold', 'silver', '#CD7F32', 'lightcoral'][:len(nomes)]
        
        bars = plt.barh(nomes, scores, color=cores)
        plt.title('Ranking dos Modelos por F1-Score', fontsize=14, fontweight='bold')
        plt.xlabel('F1-Score', fontsize=12, fontweight='bold')
        plt.ylabel('Modelos', fontsize=12, fontweight='bold')
        plt.xlim(0, 1)
        
        for bar, score in zip(bars, scores):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=11, fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('src/machine_learning/ranking_f1_score.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Radar chart do melhor modelo
        melhor_modelo = f1_sorted[0][0]
        melhor_resultados = modelos_resultados[melhor_modelo]
        
        categorias = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        valores = [
            melhor_resultados['acuracia'],
            melhor_resultados['precisao'],
            melhor_resultados['recall'],
            melhor_resultados['f1_score']
        ]
        
        angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
        valores += valores[:1]  # Fecha o círculo
        angulos += angulos[:1]
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        ax.plot(angulos, valores, 'o-', linewidth=3, label=melhor_modelo.replace('_', ' ').title())
        ax.fill(angulos, valores, alpha=0.25)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f'Perfil do Melhor Modelo\n({melhor_modelo.replace("_", " ").title()})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('src/machine_learning/radar_chart_melhor_modelo.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def gerar_relatorio_metricas(self, modelos_resultados):
        """Gera relatório completo das métricas"""
        relatorio = "=== RELATÓRIO COMPLETO DE MÉTRICAS ===\n\n"
        
        # Ranking
        f1_scores = {nome: res['f1_score'] for nome, res in modelos_resultados.items()}
        ranking = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        
        relatorio += "🏆 RANKING DOS MODELOS (F1-Score):\n"
        for i, (nome, score) in enumerate(ranking, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            relatorio += f"{emoji} {i}º lugar: {nome.replace('_', ' ').title()} - {score:.3f}\n"
        
        relatorio += "\n" + "="*60 + "\n\n"
        
        # Detalhes por modelo
        for nome, resultados in modelos_resultados.items():
            relatorio += f"📋 MODELO: {nome.replace('_', ' ').upper()}\n"
            relatorio += f"   Acurácia: {resultados['acuracia']:.3f} ({resultados['acuracia']*100:.1f}%)\n"
            relatorio += f"   Precisão: {resultados['precisao']:.3f} ({resultados['precisao']*100:.1f}%)\n"
            relatorio += f"   Recall: {resultados['recall']:.3f} ({resultados['recall']*100:.1f}%)\n"
            relatorio += f"   F1-Score: {resultados['f1_score']:.3f} ({resultados['f1_score']*100:.1f}%)\n"
            
            # Interpretação da matriz de confusão
            cm = resultados['matriz_confusao']
            tn, fp, fn, tp = cm.ravel()
            
            relatorio += f"   \n   📊 MATRIZ DE CONFUSÃO:\n"
            relatorio += f"   ┌─────────────┬─────────────┐\n"
            relatorio += f"   │ TN: {tn:7d} │ FP: {fp:7d} │\n"
            relatorio += f"   ├─────────────┼─────────────┤\n"
            relatorio += f"   │ FN: {fn:7d} │ TP: {tp:7d} │\n"
            relatorio += f"   └─────────────┴─────────────┘\n"
            
            # Interpretação prática
            total = tn + fp + fn + tp
            relatorio += f"   \n   💡 INTERPRETAÇÃO PRÁTICA:\n"
            relatorio += f"   • De {total} casos analisados:\n"
            relatorio += f"     - {tp} acidentes fatais identificados corretamente\n"
            relatorio += f"     - {tn} acidentes não fatais identificados corretamente\n"
            relatorio += f"     - {fp} falsos alarmes (predito fatal, mas não foi)\n"
            relatorio += f"     - {fn} casos perdidos (era fatal, mas não foi detectado)\n\n"
        
        # Recomendações
        melhor_modelo = ranking[0][0]
        melhor_score = ranking[0][1]
        
        relatorio += "🎯 RECOMENDAÇÕES:\n"
        relatorio += f"• Implementar o modelo {melhor_modelo.replace('_', ' ')} em produção\n"
        relatorio += f"• F1-Score de {melhor_score:.3f} indica boa capacidade preditiva\n"
        relatorio += f"• Monitorar continuamente a performance do modelo\n"
        relatorio += f"• Considerar retreinamento com novos dados periodicamente\n"
        
        if melhor_score < 0.7:
            relatorio += f"⚠️  ATENÇÃO: F1-Score abaixo de 0.7 - considerar:\n"
            relatorio += f"   • Coleta de mais dados\n"
            relatorio += f"   • Engenharia de features\n"
            relatorio += f"   • Ajuste de hiperparâmetros\n"
        
        return relatorio