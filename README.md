# 🚗 Detecção de Padrões Temporais em Acidentes Fatais

## 📋 Descrição do Projeto

Este projeto implementa uma análise completa de padrões temporais em acidentes fatais, com o objetivo de identificar horários, dias e meses críticos, bem como fatores associados (clima, tipo de pista, tipo de veículo) para reduzir a letalidade no trânsito.

## 🎯 Objetivos

- **Identificar padrões temporais** em acidentes fatais
- **Analisar fatores de risco** associados aos acidentes
- **Fornecer insights** para políticas públicas de segurança no trânsito
- **Implementar modelos preditivos** usando diferentes técnicas de regressão

## 🏆 Entrega de Valor

- **Agenda ótima de fiscalização** (álcool/velocidade)
- **Priorização de ambulâncias** e patrulhamento
- **Identificação de trechos e horários críticos**
- **Suporte à tomada de decisões** em políticas públicas

## 📊 Componentes Implementados

### 1. Análises Estatísticas Avançadas
- ✅ **Teorema Central do Limite**: Demonstração com amostras de idades
- ✅ **Análise de Correlação**: Correlações de Pearson e Spearman
- ✅ **Distribuição Normal**: Testes de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov)
- ✅ **Teste t de Student**: Comparação entre grupos
- ✅ **Teste Qui-quadrado**: Análise de independência entre variáveis categóricas

### 2. Modelos de Regressão
- ✅ **Regressão Linear Simples**: Análise univariada
- ✅ **Regressão Linear Múltipla**: Análise multivariada
- ✅ **Regressão Não Linear**: Parabólica e Exponencial
- ✅ **Regressão Bayesiana**: Com intervalos de confiança

### 3. Métodos de Otimização
- ✅ **Mínimos Quadrados Não Linear**
- ✅ **Máxima Verossimilhança**
- ✅ **Método de Gauss-Newton**
- ✅ **Algoritmo de Levenberg-Marquardt**
- ✅ **Métodos Bayesianos**

### 4. Dashboard Interativo
- ✅ **Interface web** desenvolvida em Streamlit
- ✅ **Visualizações interativas** com Plotly
- ✅ **Análises em tempo real**
- ✅ **Comparação visual** de métodos

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone ou baixe o projeto**
```bash
cd CienciaDeDados
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

### Execução

#### 1. Análise Completa (Console)
```bash
python index.py
```
Este comando executa todas as análises e gera:
- Relatório completo em texto
- Gráficos estatísticos
- Comparação de métodos de regressão

#### 2. Dashboard Interativo
```bash
streamlit run dashboard.py
```
Abre uma interface web interativa com:
- Visualizações dinâmicas
- Análises estatísticas em tempo real
- Comparação visual de métodos
- Exploração de padrões temporais

## 📁 Estrutura do Projeto

```
CienciaDeDados/
├── 📄 index.py                          # Arquivo principal
├── 📄 dashboard.py                      # Dashboard interativo
├── 📄 requirements.txt                  # Dependências
├── 📄 README.md                         # Este arquivo
├── 📁 src/                              # Módulos do projeto
│   ├── 📄 __init__.py
│   ├── 📄 xlsClass.py                   # Classe para leitura de dados
│   ├── 📄 calculosClass.py              # Cálculos estatísticos básicos
│   ├── 📄 analises_estatisticas.py      # Análises estatísticas avançadas
│   ├── 📄 regressoes.py                 # Modelos de regressão
│   └── 📄 grafico.py                    # Geração de gráficos
├── 📁 excel/                            # Dados
│   └── 📄 dados.xlsx                    # Dataset de acidentes
└── 📁 __pycache__/                      # Cache Python
```

## 📈 Resultados e Métricas

### Avaliação de Modelos
- **R²**: Coeficiente de determinação (quanto maior, melhor)
- **RMSE**: Erro quadrático médio (quanto menor, melhor)
- **Intervalos de Confiança**: Para modelos Bayesianos

### Padrões Identificados
- **Horários críticos**: Picos de acidentes em determinadas horas
- **Dias da semana**: Variações entre dias úteis e fins de semana
- **Sazonalidade**: Padrões mensais e anuais
- **Fatores climáticos**: Influência das condições meteorológicas

## 🔬 Metodologia Científica

### Teorias Estatísticas Implementadas
1. **Teorema Central do Limite**: Demonstração da convergência para normalidade
2. **Correlação**: Análise de relações lineares e monotônicas
3. **Distribuição Normal**: Verificação de normalidade dos dados
4. **Teste t de Student**: Comparação de médias entre grupos
5. **Teste Qui-quadrado**: Análise de independência entre variáveis categóricas

### Métodos de Regressão
1. **Linear**: Para relações lineares simples e múltiplas
2. **Não Linear**: Para capturar padrões complexos (parabólicos, exponenciais)
3. **Bayesiana**: Para quantificação de incerteza

## 📊 Visualizações Disponíveis

### Dashboard Interativo
- 📈 **Gráficos temporais**: Padrões por hora, dia, mês
- 🔥 **Mapas de calor**: Intensidade de acidentes
- 📊 **Comparações**: Métodos de regressão lado a lado
- 🎯 **Métricas**: R², RMSE, intervalos de confiança

### Gráficos Estáticos
- 📊 Histogramas de distribuições
- 📦 Box plots por condição
- 📈 Séries temporais
- 🔗 Matrizes de correlação

## 🎓 Aplicações Práticas

### Para Gestores Públicos
- **Otimização de recursos**: Alocação eficiente de patrulhamento
- **Planejamento de campanhas**: Horários e locais estratégicos
- **Políticas preventivas**: Baseadas em evidências estatísticas

### Para Pesquisadores
- **Metodologia robusta**: Múltiplos métodos de análise
- **Código reproduzível**: Documentado e modular
- **Extensibilidade**: Fácil adição de novos métodos

## 🔧 Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **SciPy**: Análises estatísticas
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn**: Visualizações estáticas
- **Plotly**: Visualizações interativas
- **Streamlit**: Dashboard web

## 📝 Relatórios Gerados

1. **relatorio-de-saida.txt**: Relatório básico
2. **relatorio-completo-acidentes-fatais.txt**: Relatório detalhado
3. **Gráficos PNG**: Visualizações estáticas
4. **Dashboard interativo**: Análises em tempo real

## 🤝 Contribuições

Este projeto foi desenvolvido como parte de um trabalho acadêmico sobre análise de dados e pode ser estendido com:
- Novos métodos de análise
- Diferentes datasets
- Algoritmos de machine learning avançados
- Integração com APIs de dados em tempo real

## 📞 Suporte

Para dúvidas sobre o projeto:
1. Consulte este README
2. Verifique os comentários no código
3. Execute o dashboard para exploração interativa

---

**Desenvolvido para o projeto de Ciência de Dados - Detecção de Padrões Temporais em Acidentes Fatais**