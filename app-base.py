import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Sistema de Análise de Crédito",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    .danger-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar o modelo
@st.cache_resource
def carregar_modelo():
    try:
        with open('modelo_credito.pkl', 'rb') as file:
            modelo = pickle.load(file)
        return modelo
    except FileNotFoundError:
        st.error("❌ Modelo não encontrado! Certifique-se de que o arquivo 'modelo_credito.pkl' está no repositório.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

# Função para fazer predição
def fazer_predicao(modelo, dados):
    try:
        # Probabilidade de aprovação
        probabilidade = modelo.predict_proba(dados)[0][1]
        # Decisão (1 = aprovado, 0 = negado)
        decisao = modelo.predict(dados)[0]
        return probabilidade, decisao
    except Exception as e:
        st.error(f"❌ Erro na predição: {str(e)}")
        return None, None

# Função para categorizar risco
def categorizar_risco(probabilidade):
    if probabilidade >= 0.7:
        return "Baixo Risco", "success"
    elif probabilidade >= 0.4:
        return "Risco Médio", "warning"
    else:
        return "Alto Risco", "danger"

# Função para gerar gráfico de gauge
def criar_gauge_chart(probabilidade):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probabilidade * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidade de Aprovação (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Título principal
st.markdown('<h1 class="main-header">💳 Sistema de Análise de Crédito</h1>', unsafe_allow_html=True)

# Subtítulo
st.markdown("### 🤖 Análise Inteligente com Machine Learning")

# Informações sobre o deploy
st.info("✨ **Deploy realizado por: Seu Nome** | 🚀 Powered by Streamlit Cloud")

# Carregamento do modelo
modelo = carregar_modelo()

if modelo is not None:
    # Sidebar com informações do modelo
    with st.sidebar:
        st.header("📊 Informações do Modelo")
        st.write("**Tipo:** Classificação Binária")
        st.write("**Algoritmo:** Random Forest")
        st.write("**Acurácia:** 85.2%")
        st.write("**Features:** 4 variáveis")
        
        st.header("📝 Como usar")
        st.write("1. Preencha os dados do cliente")
        st.write("2. Clique em 'Analisar Crédito'")
        st.write("3. Veja o resultado da análise")
        
        st.header("ℹ️ Sobre")
        st.write("Sistema desenvolvido para demonstração de deploy de modelos ML usando Streamlit Cloud.")

    # Formulário de entrada
    st.header("👤 Dados do Cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.slider(
            "📅 Idade",
            min_value=18,
            max_value=80,
            value=35,
            help="Idade do cliente em anos"
        )
        
        renda = st.number_input(
            "💰 Renda Mensal (R$)",
            min_value=0,
            max_value=50000,
            value=5000,
            step=500,
            help="Renda mensal bruta do cliente"
        )
    
    with col2:
        score_credito = st.slider(
            "📈 Score de Crédito",
            min_value=300,
            max_value=850,
            value=650,
            help="Score de crédito (300-850)"
        )
        
        experiencia_credito = st.selectbox(
            "🏦 Experiência com Crédito",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "Sem histórico",
                1: "Pouca experiência (1-2 anos)",
                2: "Experiência moderada (3-5 anos)",
                3: "Boa experiência (6-10 anos)",
                4: "Vasta experiência (10+ anos)"
            }[x],
            help="Experiência do cliente com produtos de crédito"
        )

    # Botão de análise
    st.markdown("---")
    
    if st.button("🔍 Analisar Crédito", type="primary", use_container_width=True):
        # Preparar dados para predição
        dados_cliente = np.array([[idade, renda, score_credito, experiencia_credito]])
        
        # Fazer predição
        probabilidade, decisao = fazer_predicao(modelo, dados_cliente)
        
        if probabilidade is not None:
            # Categorizar risco
            categoria_risco, tipo_alerta = categorizar_risco(probabilidade)
            
            st.markdown("---")
            st.header("📋 Resultado da Análise")
            
            # Métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Decisão",
                    value="✅ APROVADO" if decisao == 1 else "❌ NEGADO",
                    delta="Recomendação do modelo"
                )
            
            with col2:
                st.metric(
                    label="Probabilidade",
                    value=f"{probabilidade:.1%}",
                    delta=f"{(probabilidade - 0.5):.1%} do limite"
                )
            
            with col3:
                st.metric(
                    label="Categoria de Risco",
                    value=categoria_risco,
                    delta="Baseado na probabilidade"
                )
            
            # Gráfico de gauge
            st.subheader("📊 Visualização da Probabilidade")
            fig_gauge = criar_gauge_chart(probabilidade)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Interpretação detalhada
            st.subheader("🔍 Interpretação Detalhada")
            
            if decisao == 1:
                if probabilidade >= 0.7:
                    st.markdown("""
                    <div class="success-message">
                        <strong>✅ CRÉDITO APROVADO - BAIXO RISCO</strong><br>
                        Cliente apresenta excelente perfil para aprovação. 
                        Probabilidade alta de pagamento em dia.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-message">
                        <strong>⚠️ CRÉDITO APROVADO - RISCO MODERADO</strong><br>
                        Cliente aprovado, mas recomenda-se monitoramento. 
                        Considere limite menor ou garantias adicionais.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-message">
                    <strong>❌ CRÉDITO NEGADO - ALTO RISCO</strong><br>
                    Cliente não atende aos critérios mínimos. 
                    Alta probabilidade de inadimplência.
                </div>
                """, unsafe_allow_html=True)
            
            # Análise dos fatores
            st.subheader("📈 Análise dos Fatores")
            
            fatores_data = {
                'Fator': ['Idade', 'Renda', 'Score', 'Experiência'],
                'Valor': [idade, f"R$ {renda:,.0f}", score_credito, experiencia_credito],
                'Impacto': [
                    'Positivo' if 25 <= idade <= 55 else 'Neutro',
                    'Positivo' if renda >= 4000 else 'Negativo' if renda < 2000 else 'Neutro',
                    'Positivo' if score_credito >= 700 else 'Negativo' if score_credito < 500 else 'Neutro',
                    'Positivo' if experiencia_credito >= 2 else 'Negativo'
                ]
            }
            
            df_fatores = pd.DataFrame(fatores_data)
            
            # Colorir tabela baseado no impacto
            def color_impacto(val):
                if val == 'Positivo':
                    return 'background-color: #d4edda'
                elif val == 'Negativo':
                    return 'background-color: #f8d7da'
                else:
                    return 'background-color: #fff3cd'
            
            st.dataframe(
                df_fatores.style.applymap(color_impacto, subset=['Impacto']),
                use_container_width=True
            )
            
            # Recomendações
            st.subheader("💡 Recomendações")
            
            recomendacoes = []
            
            if decisao == 1:
                if probabilidade >= 0.8:
                    recomendacoes.extend([
                        "✅ Cliente aprovado para linha de crédito premium",
                        "✅ Considere ofertas de produtos adicionais",
                        "✅ Cliente elegível para limites mais altos"
                    ])
                elif probabilidade >= 0.6:
                    recomendacoes.extend([
                        "⚠️ Aprovação com limite conservador",
                        "⚠️ Monitoramento mensal recomendado",
                        "⚠️ Reavaliação em 6 meses"
                    ])
                else:
                    recomendacoes.extend([
                        "⚠️ Aprovação condicional",
                        "⚠️ Exigir garantias ou avalista",
                        "⚠️ Limite reduzido inicialmente"
                    ])
            else:
                recomendacoes.extend([
                    "❌ Crédito negado no momento",
                    "📚 Orientar sobre melhoria do score",
                    "🔄 Reavaliação possível em 3-6 meses"
                ])
                
                if score_credito < 500:
                    recomendacoes.append("📈 Foco prioritário: melhorar score de crédito")
                if renda < 2000:
                    recomendacoes.append("💰 Considere aumentar comprovação de renda")
            
            for rec in recomendacoes:
                st.write(f"• {rec}")
            
            # Botão para nova análise
            st.markdown("---")
            if st.button("🔄 Nova Análise", type="secondary", use_container_width=True):
                st.experimental_rerun()

    # Seção de informações adicionais
    st.markdown("---")
    st.header("📚 Informações Adicionais")
    
    with st.expander("🔍 Como funciona o modelo?"):
        st.write("""
        Este sistema utiliza um modelo de **Random Forest** treinado com dados históricos de crédito.
        
        **Variáveis consideradas:**
        - **Idade:** Impacta na estabilidade financeira
        - **Renda:** Principal indicador de capacidade de pagamento  
        - **Score de Crédito:** Histórico de comportamento financeiro
        - **Experiência:** Maturidade no uso de produtos financeiros
        
        **Processo de decisão:**
        1. Modelo calcula probabilidade de aprovação (0-100%)
        2. Threshold de 50% define aprovação/negação
        3. Categorização de risco baseada na probabilidade
        """)
    
    with st.expander("⚠️ Limitações e Considerações"):
        st.write("""
        **Este é um modelo demonstrativo para fins educacionais.**
        
        **Limitações:**
        - Modelo simplificado com apenas 4 variáveis
        - Dados sintéticos para treinamento
        - Não considera fatores externos (economia, setor, etc.)
        
        **Em produção real, considerar:**
        - Muito mais variáveis e dados externos
        - Modelos mais complexos e atualizados regularmente
        - Validação contínua e monitoramento de performance
        - Compliance com regulamentações bancárias
        """)
    
    with st.expander("🎓 Sobre este Projeto"):
        st.write("""
        **Sistema desenvolvido para a disciplina Introdução à Ciência de Dados**
        
        **Objetivos:**
        - Demonstrar deploy de modelos ML
        - Criar interface intuitiva para não-técnicos
        - Aplicar conceitos de MLOps na prática
        
        **Tecnologias utilizadas:**
        - Python + Streamlit
        - Scikit-learn para ML
        - Plotly para visualizações
        - GitHub + Streamlit Cloud para deploy
        
        **Deploy automático:** Qualquer alteração no código do GitHub atualiza automaticamente esta aplicação!
        """)

else:
    st.error("❌ Não foi possível carregar o modelo. Verifique se o arquivo 'modelo_credito.pkl' está presente no repositório.")
    st.info("💡 **Próximos passos:** Faça o upload do arquivo modelo_credito.pkl no seu repositório GitHub.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    💳 Sistema de Análise de Crédito | 🎓 Projeto Acadêmico | 
    🚀 Deploy: Streamlit Cloud | 📊 ML: Random Forest
</div>
""", unsafe_allow_html=True)