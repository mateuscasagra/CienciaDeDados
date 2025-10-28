import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.xlsClass import xlsClass
from src.analises_estatisticas import AnaliseEstatistica
from src.regressoes import AnaliseRegressao
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard - Padrões Temporais em Acidentes Fatais",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para forçar light mode
st.markdown("""
<style>
    /* Forçar tema claro em todo o app */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Sidebar em light mode */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr, section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    /* Área principal */
    .main, .css-k1vhr4, .css-18e3th9, .css-1d391kg {
        background-color: #ffffff !important;
        color: #000000 !important;
        padding-top: 1rem;
    }
    
    /* Headers e títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #ff7f0e;
        padding-left: 1rem;
    }
    
    /* Métricas */
    .stMetric {
        background-color: #f8f9fa !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    
    /* Cards e containers */
    .metric-card {
        background-color: #f8f9fa !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f0f8ff !important;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    
    /* Selectbox e inputs */
    .stSelectbox, .stMultiSelect, .stSlider {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Texto geral */
    p, span, div {
        color: #000000 !important;
    }
    
    /* Botões */
    .stButton > button {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados():
    """Carrega e prepara os dados dos acidentes."""
    try:
        leitor = xlsClass('excel/dados.xlsx')
        dados_completos = leitor.aplicaRegras()
        df = pd.DataFrame(dados_completos)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

@st.cache_data
def gerar_dados_simulados():
    """Gera dados simulados para demonstração."""
    np.random.seed(42)
    n_pontos = 1000
    
    # Simular dados temporais realistas
    horas = np.random.randint(0, 24, n_pontos)
    dias_semana = np.random.randint(1, 8, n_pontos)
    meses = np.random.randint(1, 13, n_pontos)
    
    # Padrões realistas de acidentes
    acidentes = (
        15 + 
        8 * np.sin(2 * np.pi * horas / 24) +  # Pico manhã/tarde
        5 * (dias_semana > 5) +  # Mais acidentes fins de semana
        3 * np.sin(2 * np.pi * meses / 12) +  # Variação sazonal
        np.random.normal(0, 3, n_pontos)
    )
    acidentes = np.maximum(0, acidentes)
    
    # Simular outras variáveis
    idades = np.random.normal(35, 15, n_pontos)
    idades = np.clip(idades, 18, 80)
    
    condicoes = np.random.choice(['Sol', 'Chuva', 'Nublado', 'Neblina'], n_pontos, 
                                p=[0.5, 0.2, 0.2, 0.1])
    
    tipos_veiculo = np.random.choice(['Carro', 'Moto', 'Caminhão', 'Ônibus'], n_pontos,
                                   p=[0.6, 0.25, 0.1, 0.05])
    
    return pd.DataFrame({
        'hora': horas,
        'dia_semana': dias_semana,
        'mes': meses,
        'num_acidentes': acidentes,
        'idade': idades,
        'condicao_metereologica': condicoes,
        'tipo_veiculo': tipos_veiculo,
        'tempo_sequencial': range(n_pontos)
    })

def main():
    # Configuração adicional para forçar tema claro
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    
    # Cabeçalho principal
    st.markdown('<h1 class="main-header">🚗 Dashboard - Detecção de Padrões Temporais em Acidentes Fatais</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navegação")
    opcao = st.sidebar.selectbox(
        "Escolha a análise:",
        ["Visão Geral", "Análises Estatísticas", "Regressões", "Padrões Temporais", "Comparação de Métodos"]
    )
    
    # Carregar dados
    df_original = carregar_dados()
    df_simulado = gerar_dados_simulados()
    
    if opcao == "Visão Geral":
        mostrar_visao_geral(df_original, df_simulado)
    elif opcao == "Análises Estatísticas":
        mostrar_analises_estatisticas(df_simulado)
    elif opcao == "Regressões":
        mostrar_regressoes(df_simulado)
    elif opcao == "Padrões Temporais":
        mostrar_padroes_temporais(df_simulado)
    elif opcao == "Comparação de Métodos":
        mostrar_comparacao_metodos(df_simulado)

def mostrar_visao_geral(df_original, df_simulado):
    """Mostra visão geral do projeto e dados."""
    st.markdown('<h2 class="section-header">📋 Visão Geral do Projeto</h2>', unsafe_allow_html=True)
    
    # Informações do projeto
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Objetivo do Projeto</h3>
    <p>Identificar horários, dias e meses críticos para acidentes fatais, analisando fatores associados 
    como clima, tipo de pista e tipo de veículo para reduzir a letalidade no trânsito.</p>
    
    <h3>💼 Entrega de Valor</h3>
    <ul>
        <li>Agenda ótima de fiscalização (álcool/velocidade)</li>
        <li>Priorização de ambulâncias e patrulhamento</li>
        <li>Identificação de trechos e horários críticos</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de Registros", len(df_simulado))
    
    with col2:
        st.metric("📈 Média de Acidentes/Hora", f"{df_simulado['num_acidentes'].mean():.1f}")
    
    with col3:
        st.metric("👥 Idade Média das Vítimas", f"{df_simulado['idade'].mean():.1f} anos")
    
    with col4:
        st.metric("🌧️ Condições Climáticas", len(df_simulado['condicao_metereologica'].unique()))
    
    # Gráficos de visão geral
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição de Acidentes por Hora")
        fig_hora = px.histogram(df_simulado, x='hora', y='num_acidentes', 
                               title="Acidentes por Hora do Dia",
                               labels={'hora': 'Hora do Dia', 'num_acidentes': 'Número de Acidentes'})
        fig_hora.update_layout(showlegend=False)
        st.plotly_chart(fig_hora, use_container_width=True)
    
    with col2:
        st.subheader("🌤️ Acidentes por Condição Meteorológica")
        acidentes_clima = df_simulado.groupby('condicao_metereologica')['num_acidentes'].sum().reset_index()
        fig_clima = px.pie(acidentes_clima, values='num_acidentes', names='condicao_metereologica',
                          title="Distribuição por Condição Climática")
        st.plotly_chart(fig_clima, use_container_width=True)

def mostrar_analises_estatisticas(df):
    """Mostra análises estatísticas avançadas."""
    st.markdown('<h2 class="section-header">📈 Análises Estatísticas Avançadas</h2>', unsafe_allow_html=True)
    
    # Criar instância da análise
    analise = AnaliseEstatistica(df)
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Teorema Central do Limite", 
        "🔗 Correlação", 
        "📈 Distribuição Normal", 
        "📋 Teste t-Student", 
        "🔢 Teste Qui-quadrado"
    ])
    
    with tab1:
        st.subheader("Teorema Central do Limite")
        
        # Parâmetros
        col1, col2 = st.columns(2)
        with col1:
            tamanho_amostra = st.slider("Tamanho da Amostra", 10, 100, 30)
        with col2:
            num_amostras = st.slider("Número de Amostras", 100, 2000, 1000)
        
        if st.button("Executar Análise TCL"):
            with st.spinner("Executando análise..."):
                resultado_tcl = analise.teorema_central_limite('idade', tamanho_amostra, num_amostras)
                
                if resultado_tcl:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Média Original", f"{resultado_tcl['media_original']:.2f}")
                        st.metric("Média das Médias", f"{resultado_tcl['media_das_medias']:.2f}")
                    
                    with col2:
                        st.metric("Desvio Original", f"{resultado_tcl['desvio_original']:.2f}")
                        st.metric("Erro Padrão", f"{resultado_tcl['erro_padrao_teorico']:.2f}")
                    
                    # Gráfico das médias amostrais
                    fig = px.histogram(x=resultado_tcl['medias_amostrais'], 
                                     title="Distribuição das Médias Amostrais",
                                     labels={'x': 'Média Amostral', 'y': 'Frequência'})
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Análise de Correlação")
        
        if st.button("Calcular Correlações"):
            with st.spinner("Calculando correlações..."):
                resultado_corr = analise.analise_correlacao()
                
                if 'pearson' in resultado_corr:
                    st.subheader("Matriz de Correlação de Pearson")
                    fig_corr = px.imshow(resultado_corr['pearson'], 
                                        text_auto=True, 
                                        title="Correlações de Pearson",
                                        color_continuous_scale='RdBu')
                    st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("Teste de Normalidade")
        
        variavel = st.selectbox("Escolha a variável:", ['idade', 'num_acidentes', 'hora'])
        
        if st.button("Testar Normalidade"):
            with st.spinner("Testando normalidade..."):
                resultado_norm = analise.analise_distribuicao_normal(variavel)
                
                if resultado_norm:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Média", f"{resultado_norm['media']:.2f}")
                        st.metric("Desvio Padrão", f"{resultado_norm['desvio']:.2f}")
                    
                    with col2:
                        st.metric("P-valor Shapiro", f"{resultado_norm['shapiro_p']:.4f}")
                        normalidade = "✅ Normal" if resultado_norm['normal'] else "❌ Não Normal"
                        st.metric("Resultado", normalidade)
                    
                    # Histograma com curva normal
                    dados_var = df[variavel].dropna()
                    fig = go.Figure()
                    
                    # Histograma
                    fig.add_trace(go.Histogram(x=dados_var, name="Dados", opacity=0.7))
                    
                    # Curva normal teórica
                    x_norm = np.linspace(dados_var.min(), dados_var.max(), 100)
                    y_norm = len(dados_var) * (x_norm[1] - x_norm[0]) * \
                            (1/(resultado_norm['desvio'] * np.sqrt(2*np.pi))) * \
                            np.exp(-0.5 * ((x_norm - resultado_norm['media'])/resultado_norm['desvio'])**2)
                    
                    fig.add_trace(go.Scatter(x=x_norm, y=y_norm, 
                                           name="Curva Normal Teórica", 
                                           line=dict(color='red', width=2)))
                    
                    fig.update_layout(title=f"Distribuição de {variavel} vs Normal Teórica")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Teste t de Student")
        
        if st.button("Executar Teste t"):
            with st.spinner("Executando teste t..."):
                resultado_t = analise.teste_t_student('idade', 'condicao_metereologica')
                
                if resultado_t:
                    st.success("Teste t executado com sucesso!")
                    st.info("Verifique o console para resultados detalhados.")
    
    with tab5:
        st.subheader("Teste Qui-quadrado")
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Primeira variável:", ['condicao_metereologica', 'tipo_veiculo'])
        with col2:
            var2 = st.selectbox("Segunda variável:", ['tipo_veiculo', 'condicao_metereologica'])
        
        if st.button("Executar Teste Qui-quadrado"):
            with st.spinner("Executando teste..."):
                resultado_qui = analise.teste_qui_quadrado(var1, var2)
                
                if resultado_qui:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Estatística χ²", f"{resultado_qui['chi2_stat']:.4f}")
                        st.metric("P-valor", f"{resultado_qui['p_valor']:.4f}")
                    
                    with col2:
                        st.metric("Graus de Liberdade", resultado_qui['dof'])
                        dependencia = "✅ Dependentes" if resultado_qui['dependentes'] else "❌ Independentes"
                        st.metric("Resultado", dependencia)
                    
                    # Tabela de contingência
                    st.subheader("Tabela de Contingência")
                    st.dataframe(resultado_qui['tabela_contingencia'])

def mostrar_regressoes(df):
    """Mostra análises de regressão."""
    st.markdown('<h2 class="section-header">📊 Análises de Regressão</h2>', unsafe_allow_html=True)
    
    # Criar instância da análise
    regressao = AnaliseRegressao(df)
    
    # Tabs para diferentes tipos de regressão
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Linear Simples", 
        "📊 Linear Múltipla", 
        "🔄 Não Linear", 
        "🎯 Bayesiana"
    ])
    
    with tab1:
        st.subheader("Regressão Linear Simples")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variável X:", ['hora', 'dia_semana', 'mes', 'idade'])
        with col2:
            y_var = st.selectbox("Variável Y:", ['num_acidentes', 'idade'])
        
        if st.button("Executar Regressão Linear Simples"):
            with st.spinner("Executando regressão..."):
                resultado = regressao.regressao_linear_simples(x_var, y_var, df)
                
                if resultado:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R² (Teste)", f"{resultado['r2_test']:.4f}")
                    with col2:
                        st.metric("RMSE (Teste)", f"{resultado['rmse_test']:.4f}")
                    with col3:
                        st.metric("Coeficiente", f"{resultado['coeficiente']:.4f}")
                    
                    # Gráfico de dispersão com linha de regressão
                    fig = go.Figure()
                    
                    # Pontos de teste
                    fig.add_trace(go.Scatter(x=resultado['X_test'].flatten(), 
                                           y=resultado['y_test'],
                                           mode='markers', 
                                           name='Dados de Teste',
                                           opacity=0.6))
                    
                    # Linha de regressão
                    fig.add_trace(go.Scatter(x=resultado['X_test'].flatten(), 
                                           y=resultado['y_pred_test'],
                                           mode='lines', 
                                           name='Linha de Regressão',
                                           line=dict(color='red', width=2)))
                    
                    fig.update_layout(title=f"Regressão Linear: {y_var} vs {x_var}",
                                    xaxis_title=x_var,
                                    yaxis_title=y_var)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Regressão Linear Múltipla")
        
        variaveis_x = st.multiselect("Variáveis X:", ['hora', 'dia_semana', 'mes'], 
                                   default=['hora', 'dia_semana'])
        y_var = st.selectbox("Variável Y:", ['num_acidentes'], key='multi_y')
        
        if st.button("Executar Regressão Múltipla"):
            if len(variaveis_x) >= 2:
                with st.spinner("Executando regressão múltipla..."):
                    resultado = regressao.regressao_linear_multipla(variaveis_x, y_var, df)
                    
                    if resultado:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("R² (Teste)", f"{resultado['r2_test']:.4f}")
                            st.metric("RMSE (Teste)", f"{resultado['rmse_test']:.4f}")
                        
                        with col2:
                            st.subheader("Coeficientes")
                            for i, var in enumerate(variaveis_x):
                                st.write(f"{var}: {resultado['coeficientes'][i]:.4f}")
            else:
                st.warning("Selecione pelo menos 2 variáveis X.")
    
    with tab3:
        st.subheader("Regressão Não Linear")
        
        tipo_funcao = st.selectbox("Tipo de Função:", ['Parabólica', 'Exponencial'])
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variável X:", ['tempo_sequencial', 'hora'], key='nonlinear_x')
        with col2:
            y_var = st.selectbox("Variável Y:", ['num_acidentes'], key='nonlinear_y')
        
        if st.button("Executar Regressão Não Linear"):
            with st.spinner("Executando regressão não linear..."):
                if tipo_funcao == 'Parabólica':
                    resultado = regressao.regressao_nao_linear_parabola(x_var, y_var, df)
                else:
                    resultado = regressao.regressao_nao_linear_exponencial(x_var, y_var, df)
                
                if resultado and 'metodos' in resultado:
                    st.subheader("Comparação de Métodos")
                    
                    metodos_df = []
                    for metodo, dados in resultado['metodos'].items():
                        metodos_df.append({
                            'Método': metodo,
                            'R²': dados['r2'],
                            'RMSE': dados['rmse']
                        })
                    
                    if metodos_df:
                        df_metodos = pd.DataFrame(metodos_df)
                        st.dataframe(df_metodos)
                        
                        # Gráfico dos resultados
                        fig = go.Figure()
                        
                        # Dados originais
                        fig.add_trace(go.Scatter(x=resultado['x'], y=resultado['y'],
                                               mode='markers', name='Dados Originais',
                                               opacity=0.6))
                        
                        # Predições dos diferentes métodos
                        cores = ['red', 'blue', 'green', 'orange']
                        for i, (metodo, dados) in enumerate(resultado['metodos'].items()):
                            if 'y_pred' in dados:
                                fig.add_trace(go.Scatter(x=resultado['x'], y=dados['y_pred'],
                                                       mode='lines', name=f'{metodo}',
                                                       line=dict(color=cores[i % len(cores)])))
                        
                        fig.update_layout(title=f"Regressão {tipo_funcao}: {y_var} vs {x_var}")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Regressão Bayesiana")
        
        variaveis_x = st.multiselect("Variáveis X:", ['hora', 'dia_semana', 'mes'], 
                                   default=['hora', 'dia_semana'], key='bayes_x')
        y_var = st.selectbox("Variável Y:", ['num_acidentes'], key='bayes_y')
        
        if st.button("Executar Regressão Bayesiana"):
            if len(variaveis_x) >= 1:
                with st.spinner("Executando regressão Bayesiana..."):
                    resultado = regressao.regressao_bayesiana(variaveis_x, y_var, df)
                    
                    if resultado:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("R² (Teste)", f"{resultado['r2_test']:.4f}")
                        with col2:
                            st.metric("RMSE (Teste)", f"{resultado['rmse_test']:.4f}")
                        with col3:
                            st.metric("Incerteza Média", f"{resultado['incerteza_media']:.4f}")
                        
                        # Gráfico com intervalos de confiança
                        fig = go.Figure()
                        
                        indices = range(len(resultado['y_pred_test']))
                        
                        # Intervalo de confiança
                        fig.add_trace(go.Scatter(
                            x=list(indices) + list(indices)[::-1],
                            y=list(resultado['y_pred_test'] + 2*resultado['y_std_test']) + 
                              list(resultado['y_pred_test'] - 2*resultado['y_std_test'])[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Intervalo de Confiança (95%)'
                        ))
                        
                        # Predições
                        fig.add_trace(go.Scatter(x=indices, y=resultado['y_pred_test'],
                                               mode='lines', name='Predições',
                                               line=dict(color='blue')))
                        
                        fig.update_layout(title="Regressão Bayesiana com Intervalos de Confiança")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Selecione pelo menos 1 variável X.")

def mostrar_padroes_temporais(df):
    """Mostra análise de padrões temporais."""
    st.markdown('<h2 class="section-header">⏰ Padrões Temporais em Acidentes</h2>', unsafe_allow_html=True)
    
    # Análise por hora
    st.subheader("📊 Padrões por Hora do Dia")
    acidentes_hora = df.groupby('hora')['num_acidentes'].mean().reset_index()
    
    fig_hora = px.line(acidentes_hora, x='hora', y='num_acidentes',
                      title="Média de Acidentes por Hora do Dia",
                      markers=True)
    fig_hora.update_layout(xaxis_title="Hora do Dia", yaxis_title="Média de Acidentes")
    st.plotly_chart(fig_hora, use_container_width=True)
    
    # Análise por dia da semana
    st.subheader("📅 Padrões por Dia da Semana")
    dias_nomes = {1: 'Segunda', 2: 'Terça', 3: 'Quarta', 4: 'Quinta', 
                  5: 'Sexta', 6: 'Sábado', 7: 'Domingo'}
    
    acidentes_dia = df.groupby('dia_semana')['num_acidentes'].mean().reset_index()
    acidentes_dia['dia_nome'] = acidentes_dia['dia_semana'].map(dias_nomes)
    
    fig_dia = px.bar(acidentes_dia, x='dia_nome', y='num_acidentes',
                    title="Média de Acidentes por Dia da Semana")
    st.plotly_chart(fig_dia, use_container_width=True)
    
    # Heatmap hora vs dia da semana
    st.subheader("🔥 Mapa de Calor: Hora vs Dia da Semana")
    
    pivot_data = df.pivot_table(values='num_acidentes', 
                               index='hora', 
                               columns='dia_semana', 
                               aggfunc='mean')
    
    fig_heatmap = px.imshow(pivot_data, 
                           title="Intensidade de Acidentes por Hora e Dia",
                           labels=dict(x="Dia da Semana", y="Hora do Dia", color="Acidentes"),
                           color_continuous_scale="Reds")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Análise sazonal
    st.subheader("🌍 Padrões Sazonais")
    meses_nomes = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                   7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    
    acidentes_mes = df.groupby('mes')['num_acidentes'].mean().reset_index()
    acidentes_mes['mes_nome'] = acidentes_mes['mes'].map(meses_nomes)
    
    fig_mes = px.line(acidentes_mes, x='mes_nome', y='num_acidentes',
                     title="Média de Acidentes por Mês",
                     markers=True)
    fig_mes.update_layout(xaxis_title="Mês", yaxis_title="Média de Acidentes")
    st.plotly_chart(fig_mes, use_container_width=True)

def mostrar_comparacao_metodos(df):
    """Mostra comparação entre métodos de regressão."""
    st.markdown('<h2 class="section-header">⚖️ Comparação de Métodos de Regressão</h2>', unsafe_allow_html=True)
    
    if st.button("Executar Comparação Completa"):
        with st.spinner("Executando todas as análises..."):
            regressao = AnaliseRegressao(df)
            comparacao = regressao.comparar_metodos()
            
            if comparacao:
                # Criar DataFrame para visualização
                df_comp = pd.DataFrame([
                    {
                        'Método': dados['tipo'],
                        'R²': dados['r2'],
                        'RMSE': dados['rmse'],
                        'Incerteza': dados.get('incerteza', 0)
                    }
                    for dados in comparacao.values()
                ])
                
                # Tabela de comparação
                st.subheader("📊 Tabela de Comparação")
                st.dataframe(df_comp.style.highlight_max(subset=['R²']).highlight_min(subset=['RMSE']))
                
                # Gráficos de comparação
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_r2 = px.bar(df_comp, x='Método', y='R²',
                                   title="Comparação de R² por Método",
                                   color='R²', color_continuous_scale='Blues')
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    fig_rmse = px.bar(df_comp, x='Método', y='RMSE',
                                     title="Comparação de RMSE por Método",
                                     color='RMSE', color_continuous_scale='Reds')
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Recomendações
                melhor_r2 = df_comp.loc[df_comp['R²'].idxmax()]
                melhor_rmse = df_comp.loc[df_comp['RMSE'].idxmin()]
                
                st.subheader("🎯 Recomendações")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Melhor R²:** {melhor_r2['Método']} ({melhor_r2['R²']:.4f})")
                
                with col2:
                    st.success(f"**Menor RMSE:** {melhor_rmse['Método']} ({melhor_rmse['RMSE']:.4f})")
                
                # Análise detalhada
                st.markdown("""
                <div class="info-box">
                <h3>📋 Interpretação dos Resultados</h3>
                <ul>
                    <li><strong>R²:</strong> Indica a proporção da variância explicada pelo modelo (quanto maior, melhor)</li>
                    <li><strong>RMSE:</strong> Erro quadrático médio (quanto menor, melhor)</li>
                    <li><strong>Regressão Bayesiana:</strong> Fornece intervalos de confiança para as predições</li>
                    <li><strong>Regressão Não Linear:</strong> Pode capturar padrões mais complexos nos dados</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()