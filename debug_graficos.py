import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

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

def testar_graficos():
    """Testa a geração de gráficos."""
    st.title("🔍 Debug dos Gráficos")
    
    # Gerar dados
    df = gerar_dados_simulados()
    
    st.write("### Informações dos Dados:")
    st.write(f"Shape: {df.shape}")
    st.write(f"Colunas: {list(df.columns)}")
    st.write("### Primeiras 5 linhas:")
    st.dataframe(df.head())
    
    st.write("### Estatísticas básicas:")
    st.write(df.describe())
    
    # Teste 1: Gráfico por hora
    st.write("### Teste 1: Padrões por Hora")
    acidentes_hora = df.groupby('hora')['num_acidentes'].mean().reset_index()
    st.write(f"Dados agrupados por hora: {acidentes_hora.shape}")
    st.dataframe(acidentes_hora.head())
    
    if not acidentes_hora.empty:
        fig_hora = px.line(acidentes_hora, x='hora', y='num_acidentes',
                          title="Média de Acidentes por Hora do Dia",
                          markers=True)
        fig_hora.update_layout(xaxis_title="Hora do Dia", yaxis_title="Média de Acidentes")
        st.plotly_chart(fig_hora, use_container_width=True)
    else:
        st.error("Dados por hora estão vazios!")
    
    # Teste 2: Gráfico por dia da semana
    st.write("### Teste 2: Padrões por Dia da Semana")
    dias_nomes = {1: 'Segunda', 2: 'Terça', 3: 'Quarta', 4: 'Quinta', 
                  5: 'Sexta', 6: 'Sábado', 7: 'Domingo'}
    
    acidentes_dia = df.groupby('dia_semana')['num_acidentes'].mean().reset_index()
    acidentes_dia['dia_nome'] = acidentes_dia['dia_semana'].map(dias_nomes)
    st.write(f"Dados agrupados por dia: {acidentes_dia.shape}")
    st.dataframe(acidentes_dia)
    
    if not acidentes_dia.empty:
        fig_dia = px.bar(acidentes_dia, x='dia_nome', y='num_acidentes',
                        title="Média de Acidentes por Dia da Semana")
        st.plotly_chart(fig_dia, use_container_width=True)
    else:
        st.error("Dados por dia estão vazios!")
    
    # Teste 3: Heatmap
    st.write("### Teste 3: Mapa de Calor")
    try:
        pivot_data = df.pivot_table(values='num_acidentes', 
                                   index='hora', 
                                   columns='dia_semana', 
                                   aggfunc='mean')
        st.write(f"Pivot table shape: {pivot_data.shape}")
        st.write("Pivot table:")
        st.dataframe(pivot_data)
        
        if not pivot_data.empty:
            fig_heatmap = px.imshow(pivot_data, 
                                   title="Intensidade de Acidentes por Hora e Dia",
                                   labels=dict(x="Dia da Semana", y="Hora do Dia", color="Acidentes"),
                                   color_continuous_scale="Reds")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.error("Pivot table está vazia!")
    except Exception as e:
        st.error(f"Erro no heatmap: {e}")

if __name__ == "__main__":
    testar_graficos()