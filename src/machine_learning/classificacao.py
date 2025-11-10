import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AnaliseClassificacao:
    def __init__(self, dados):
        self.dados = dados
        self.df = pd.DataFrame(dados)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.modelos = {}
        self.resultados = {}
        
    def preparar_dados(self):
        """Prepara os dados para classificação"""
        # Remove registros com valores nulos
        df_clean = self.df.dropna()
        
        # Variáveis preditoras
        features = ['fase_dia', 'condicao_metereologica', 'causa_acidente', 'tipo_acidente', 'idade', 'sexo']
        
        # Variável alvo (mortos: 0 = não fatal, 1 = fatal)
        target = 'mortos'
        
        X = df_clean[features].copy()
        y = df_clean[target].copy()
        
        # Codifica variáveis categóricas
        for coluna in ['fase_dia', 'condicao_metereologica', 'causa_acidente', 'tipo_acidente', 'sexo']:
            if coluna in X.columns:
                le = LabelEncoder()
                X[coluna] = le.fit_transform(X[coluna].astype(str))
                self.label_encoders[coluna] = le
        
        return X, y
    
    def treinar_modelos(self):
        """Treina todos os modelos de classificação"""
        X, y = self.preparar_dados()
        
        # Divide dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Normaliza dados para KNN e Rede Neural
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Árvore de Decisão
        self.modelos['arvore'] = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.modelos['arvore'].fit(X_train, y_train)
        
        # 2. Random Forest
        self.modelos['random_forest'] = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.modelos['random_forest'].fit(X_train, y_train)
        
        # 3. KNN
        self.modelos['knn'] = KNeighborsClassifier(n_neighbors=5)
        self.modelos['knn'].fit(X_train_scaled, y_train)
        
        # 4. Rede Neural
        self.modelos['rede_neural'] = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
        self.modelos['rede_neural'].fit(X_train_scaled, y_train)
        
        # Avalia todos os modelos
        for nome, modelo in self.modelos.items():
            if nome in ['knn', 'rede_neural']:
                y_pred = modelo.predict(X_test_scaled)
            else:
                y_pred = modelo.predict(X_test)
            
            self.resultados[nome] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'acuracia': accuracy_score(y_test, y_pred),
                'precisao': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'matriz_confusao': confusion_matrix(y_test, y_pred)
            }
        
        return X_train, X_test, y_train, y_test
    
    def gerar_graficos_classificacao(self):
        """Gera gráficos para visualizar os resultados da classificação"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análise de Classificação - Predição de Acidentes Fatais', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Comparação de métricas
        metricas = ['acuracia', 'precisao', 'recall', 'f1_score']
        modelos_nomes = list(self.resultados.keys())
        
        dados_metricas = []
        for metrica in metricas:
            valores = [self.resultados[modelo][metrica] for modelo in modelos_nomes]
            dados_metricas.append(valores)
        
        x = np.arange(len(modelos_nomes))
        width = 0.2
        
        for i, metrica in enumerate(metricas):
            axes[0, 0].bar(x + i*width, dados_metricas[i], width, label=metrica.capitalize())
        
        axes[0, 0].set_title('Comparação de Métricas por Modelo')
        axes[0, 0].set_xlabel('Modelos')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels([nome.replace('_', ' ').title() for nome in modelos_nomes])
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # Gráfico 2: Matriz de Confusão - Melhor Modelo
        melhor_modelo = max(self.resultados.keys(), key=lambda x: self.resultados[x]['f1_score'])
        cm = self.resultados[melhor_modelo]['matriz_confusao']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'Matriz de Confusão - {melhor_modelo.replace("_", " ").title()}')
        axes[0, 1].set_xlabel('Predito')
        axes[0, 1].set_ylabel('Real')
        
        # Gráfico 3: Importância das Features (Random Forest)
        if 'random_forest' in self.modelos:
            importancias = self.modelos['random_forest'].feature_importances_
            features = ['Fase Dia', 'Clima', 'Causa', 'Tipo Acidente', 'Idade', 'Sexo']
            
            indices = np.argsort(importancias)[::-1]
            axes[0, 2].bar(range(len(importancias)), importancias[indices])
            axes[0, 2].set_title('Importância das Variáveis (Random Forest)')
            axes[0, 2].set_xlabel('Variáveis')
            axes[0, 2].set_ylabel('Importância')
            axes[0, 2].set_xticks(range(len(importancias)))
            axes[0, 2].set_xticklabels([features[i] for i in indices], rotation=45)
        
        # Gráfico 4: Árvore de Decisão (simplificada)
        if 'arvore' in self.modelos:
            plot_tree(self.modelos['arvore'], max_depth=3, filled=True, 
                     feature_names=['Fase Dia', 'Clima', 'Causa', 'Tipo Acidente', 'Idade', 'Sexo'],
                     class_names=['Não Fatal', 'Fatal'], ax=axes[1, 0])
            axes[1, 0].set_title('Árvore de Decisão (Simplificada)')
        
        # Gráfico 5: Ranking dos Modelos
        ranking_f1 = sorted(self.resultados.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        nomes_ranking = [nome.replace('_', ' ').title() for nome, _ in ranking_f1]
        scores_ranking = [dados['f1_score'] for _, dados in ranking_f1]
        
        cores = ['gold', 'silver', '#CD7F32', 'lightcoral']  # Ouro, Prata, Bronze, Coral
        axes[1, 1].barh(nomes_ranking, scores_ranking, color=cores[:len(nomes_ranking)])
        axes[1, 1].set_title('Ranking dos Modelos (F1-Score)')
        axes[1, 1].set_xlabel('F1-Score')
        
        # Gráfico 6: Distribuição de Predições
        melhor_y_pred = self.resultados[melhor_modelo]['y_pred']
        melhor_y_test = self.resultados[melhor_modelo]['y_test']
        
        dados_pred = pd.DataFrame({
            'Real': melhor_y_test,
            'Predito': melhor_y_pred
        })
        
        pred_counts = dados_pred.groupby(['Real', 'Predito']).size().unstack(fill_value=0)
        pred_counts.plot(kind='bar', ax=axes[1, 2], stacked=True)
        axes[1, 2].set_title(f'Distribuição de Predições - {melhor_modelo.replace("_", " ").title()}')
        axes[1, 2].set_xlabel('Valor Real')
        axes[1, 2].set_ylabel('Quantidade')
        axes[1, 2].legend(['Predito: Não Fatal', 'Predito: Fatal'])
        
        plt.tight_layout()
        plt.savefig('classificacao_acidentes_fatais.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # AGORA GERA CADA GRÁFICO INDIVIDUALMENTE
        self.gerar_graficos_classificacao_individuais()
        
        return fig
    
    def gerar_graficos_classificacao_individuais(self):
        """Gera cada gráfico de classificação individualmente para uso nos slides"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        melhor_modelo = max(self.resultados.keys(), key=lambda x: self.resultados[x]['f1_score'])
        
        # 1. Comparação de Métricas (barras agrupadas)
        plt.figure(figsize=(12, 8))
        modelos_nomes = list(self.resultados.keys())
        metricas = ['acuracia', 'precisao', 'recall', 'f1_score']
        
        x = np.arange(len(modelos_nomes))
        width = 0.2
        
        for i, metrica in enumerate(metricas):
            valores = [self.resultados[modelo][metrica] for modelo in modelos_nomes]
            plt.bar(x + i*width, valores, width, label=metrica.capitalize())
        
        plt.title('Comparação de Métricas por Modelo', fontsize=14, fontweight='bold')
        plt.xlabel('Modelos', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x + width * 1.5, [nome.replace('_', ' ').title() for nome in modelos_nomes], rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('src/machine_learning/comparacao_metricas_modelos.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Matriz de Confusão do Melhor Modelo
        plt.figure(figsize=(10, 8))
        cm = self.resultados[melhor_modelo]['matriz_confusao']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Não Fatal', 'Fatal'],
                   yticklabels=['Não Fatal', 'Fatal'],
                   cbar_kws={'label': 'Quantidade'})
        plt.title(f'Matriz de Confusão - {melhor_modelo.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predito', fontsize=12, fontweight='bold')
        plt.ylabel('Real', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('src/machine_learning/matriz_confusao_melhor_modelo.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Ranking dos Modelos (F1-Score)
        plt.figure(figsize=(10, 8))
        ranking_f1 = sorted(self.resultados.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        nomes_ranking = [nome.replace('_', ' ').title() for nome, _ in ranking_f1]
        scores_ranking = [dados['f1_score'] for _, dados in ranking_f1]
        
        cores = ['gold', 'silver', '#CD7F32', 'lightcoral']
        bars = plt.barh(nomes_ranking, scores_ranking, color=cores[:len(nomes_ranking)])
        plt.title('Ranking dos Modelos (F1-Score)', fontsize=14, fontweight='bold')
        plt.xlabel('F1-Score', fontsize=12, fontweight='bold')
        plt.ylabel('Modelos', fontsize=12, fontweight='bold')
        plt.xlim(0, 1)
        
        # Adiciona valores nas barras
        for bar, score in zip(bars, scores_ranking):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('src/machine_learning/ranking_modelos_f1.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Importância das Features (Random Forest) - se disponível
        if 'random_forest' in self.modelos:
            plt.figure(figsize=(10, 8))
            importancias = self.modelos['random_forest'].feature_importances_
            features = ['Fase Dia', 'Clima', 'Causa', 'Tipo Acidente', 'Idade', 'Sexo']
            
            indices = np.argsort(importancias)[::-1]
            cores = plt.cm.viridis(np.linspace(0, 1, len(importancias)))
            bars = plt.barh(range(len(importancias)), importancias[indices], color=cores)
            plt.yticks(range(len(importancias)), [features[i] for i in indices])
            plt.xlabel('Importância', fontsize=12, fontweight='bold')
            plt.ylabel('Variáveis', fontsize=12, fontweight='bold')
            plt.title('Importância das Variáveis (Random Forest)', fontsize=14, fontweight='bold')
            
            # Adiciona valores
            for i, (bar, imp) in enumerate(zip(bars, importancias[indices])):
                plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{imp:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
            
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('src/machine_learning/importancia_features_rf.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def gerar_relatorio_classificacao(self):
        """Gera relatório detalhado dos resultados de classificação"""
        relatorio = "=== RELATÓRIO DE CLASSIFICAÇÃO - PREDIÇÃO DE ACIDENTES FATAIS ===\n\n"
        
        # Ranking dos modelos
        ranking = sorted(self.resultados.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        relatorio += "RANKING DOS MODELOS (por F1-Score):\n"
        for i, (nome, dados) in enumerate(ranking, 1):
            relatorio += f"{i}º {nome.replace('_', ' ').title()}: {dados['f1_score']:.3f}\n"
        
        relatorio += "\n" + "="*60 + "\n\n"
        
        # Detalhes de cada modelo
        for nome, dados in self.resultados.items():
            relatorio += f"MODELO: {nome.replace('_', ' ').upper()}\n"
            relatorio += f"  • Acurácia: {dados['acuracia']:.3f} ({dados['acuracia']*100:.1f}%)\n"
            relatorio += f"  • Precisão: {dados['precisao']:.3f} ({dados['precisao']*100:.1f}%)\n"
            relatorio += f"  • Recall: {dados['recall']:.3f} ({dados['recall']*100:.1f}%)\n"
            relatorio += f"  • F1-Score: {dados['f1_score']:.3f} ({dados['f1_score']*100:.1f}%)\n"
            
            # Matriz de confusão
            cm = dados['matriz_confusao']
            relatorio += f"  • Matriz de Confusão:\n"
            relatorio += f"    Verdadeiros Negativos: {cm[0,0]} | Falsos Positivos: {cm[0,1]}\n"
            relatorio += f"    Falsos Negativos: {cm[1,0]} | Verdadeiros Positivos: {cm[1,1]}\n\n"
        
        # Melhor modelo
        melhor_modelo = ranking[0][0]
        melhor_dados = ranking[0][1]
        
        relatorio += f"🏆 MELHOR MODELO: {melhor_modelo.replace('_', ' ').upper()}\n"
        relatorio += f"   F1-Score: {melhor_dados['f1_score']:.3f} ({melhor_dados['f1_score']*100:.1f}%)\n"
        relatorio += f"   Acurácia: {melhor_dados['acuracia']:.3f} ({melhor_dados['acuracia']*100:.1f}%)\n\n"
        
        # Importância das variáveis (se Random Forest for o melhor)
        if 'random_forest' in self.modelos:
            importancias = self.modelos['random_forest'].feature_importances_
            features = ['Fase do Dia', 'Condição Meteorológica', 'Causa do Acidente', 'Tipo de Acidente', 'Idade', 'Sexo']
            
            relatorio += "IMPORTÂNCIA DAS VARIÁVEIS (Random Forest):\n"
            for i, (feature, imp) in enumerate(zip(features, importancias), 1):
                relatorio += f"  {i}º {feature}: {imp:.3f} ({imp*100:.1f}%)\n"
            relatorio += "\n"
        
        relatorio += "RECOMENDAÇÕES PARA POLÍTICAS PÚBLICAS:\n"
        relatorio += f"• Usar o modelo {melhor_modelo.replace('_', ' ')} para predição de risco\n"
        relatorio += f"• Acurácia de {melhor_dados['acuracia']*100:.1f}% na identificação de acidentes fatais\n"
        relatorio += "• Focar nas variáveis mais importantes identificadas pelo Random Forest\n"
        relatorio += "• Implementar sistema de alerta baseado nas predições do modelo\n"
        
        return relatorio
    
    def obter_regras_arvore(self):
        """Extrai regras interpretáveis da árvore de decisão"""
        if 'arvore' not in self.modelos:
            return "Árvore de decisão não foi treinada."
        
        arvore = self.modelos['arvore']
        features = ['Fase do Dia', 'Condição Meteorológica', 'Causa do Acidente', 'Tipo de Acidente', 'Idade', 'Sexo']
        
        def extrair_regras(tree, feature_names, node=0, depth=0, parent_rule=""):
            regras = []
            
            if tree.children_left[node] == tree.children_right[node]:  # Folha
                classe = "Fatal" if tree.value[node][0][1] > tree.value[node][0][0] else "Não Fatal"
                confianca = max(tree.value[node][0]) / sum(tree.value[node][0])
                regras.append(f"{'  ' * depth}→ {classe} (confiança: {confianca:.2f})")
            else:
                feature = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Regra esquerda
                regra_esq = f"{feature} ≤ {threshold:.1f}"
                regras.append(f"{'  ' * depth}SE {regra_esq}:")
                regras.extend(extrair_regras(tree, feature_names, tree.children_left[node], depth + 1))
                
                # Regra direita
                regra_dir = f"{feature} > {threshold:.1f}"
                regras.append(f"{'  ' * depth}SENÃO SE {regra_dir}:")
                regras.extend(extrair_regras(tree, feature_names, tree.children_right[node], depth + 1))
            
            return regras
        
        regras = extrair_regras(arvore.tree_, features)
        return "\n".join(regras)