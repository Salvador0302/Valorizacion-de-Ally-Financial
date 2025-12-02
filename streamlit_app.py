"""
Panel de Valoración de Ally Financial
===================================
Panel interactivo en Streamlit para el análisis de valoración de acciones.

Ejecutar con: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, '.')

from src.data_loader import DataLoader
from src.valuation import ValuationEngine
from src.monte_carlo import MonteCarloSimulation
from src.sec_analyzer import SECAnalyzer, format_report_for_display
from src.chatbot import ValuationChatbot

# Configuración de la página
st.set_page_config(
    page_title="Grupo 8 - Monografía 2 | Panel de Valoración ALLY",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-buy {
        color: #28a745;
        font-weight: bold;
    }
    .recommendation-sell {
        color: #dc3545;
        font-weight: bold;
    }
    .recommendation-hold {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 0;">Grupo 8 - Monografía 2</p>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">📊 Panel de Valoración de Ally Financial (ALLY)</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Selección de ticker (extensible)
ticker = st.sidebar.text_input("Ticker", value="ALLY", max_chars=5)

# Valuation parameters
st.sidebar.subheader("📊 Parámetros de Valoración")
required_return = st.sidebar.slider("Rentabilidad requerida (WACC)", 0.05, 0.20, 0.10, 0.01, format="%.2f")
growth_rate = st.sidebar.slider("Tasa de crecimiento (Etapa 1)", 0.01, 0.15, 0.05, 0.01, format="%.2f")
terminal_growth = st.sidebar.slider("Tasa de crecimiento terminal", 0.01, 0.05, 0.02, 0.005, format="%.3f")
forecast_years = st.sidebar.slider("Años de pronóstico", 3, 10, 5)

# Monte Carlo parameters
st.sidebar.subheader("🎲 Parámetros Monte Carlo")
n_simulations = st.sidebar.selectbox("Número de simulaciones", [1000, 5000, 10000, 50000], index=2)
mc_days = st.sidebar.selectbox("Días de pronóstico", [63, 126, 252, 504], index=2, 
                                format_func=lambda x: f"{x} días ({x//252} año{'s' if x//252 > 1 else ''})" if x >= 252 else f"{x} días ({x//21} meses)")

# Initialize data with caching
@st.cache_data(ttl=3600)
def load_data(ticker_symbol):
    """Cargar y cachear datos financieros.

    Nota: No devolvemos el objeto `DataLoader` porque no es serializable por
    `st.cache_data`. Sólo devolvemos estructuras serializables (DataFrame/dict).
    Si necesitas cachear recursos no serializables, usa `st.cache_resource`.
    """
    loader = DataLoader(ticker=ticker_symbol)
    prices = loader.get_historical_prices(period="5y")
    summary = loader.get_summary()
    return prices, summary

@st.cache_data(ttl=3600)
def run_valuations(ticker_symbol, req_return, growth, term_growth, years):
    """Ejecutar y cachear los cálculos de valoración."""
    loader = DataLoader(ticker=ticker_symbol)
    valuation = ValuationEngine(data_loader=loader)
    
    book_value = valuation.book_value_valuation()
    adj_book = valuation.adjusted_book_value_valuation()
    pe_val = valuation.pe_ratio_valuation()
    ddm = valuation.dividend_discount_model(
        required_return=req_return,
        growth_rate=growth,
        terminal_growth=term_growth,
        forecast_years=years
    )
    comparables = valuation.comparable_companies_valuation()
    dcf = valuation.dcf_fcf_valuation(
        required_return=req_return,
        growth_rate=growth,
        terminal_growth=term_growth,
        forecast_years=years
    )
    
    summary_df = valuation.get_valuation_summary()
    fair_value = valuation.get_fair_value_estimate()
    
    return {
        'book_value': book_value,
        'adj_book': adj_book,
        'pe_val': pe_val,
        'ddm': ddm,
        'comparables': comparables,
        'dcf': dcf,
        'summary': summary_df,
        'fair_value': fair_value
    }

@st.cache_data(ttl=3600)
def run_monte_carlo(prices_data, n_sims, days):
    """Ejecutar y cachear la simulación Monte Carlo."""
    mc = MonteCarloSimulation(n_simulations=n_sims, n_days=days, random_seed=42)
    return mc.run_simulation(prices_data)

# Load data
try:
    with st.spinner("Cargando datos financieros..."):
        prices, summary = load_data(ticker)
    
    current_price = summary.get('current_price', 0)
    
    # Run valuations primero
    with st.spinner("Ejecutando modelos de valoración..."):
        valuations = run_valuations(ticker, required_return, growth_rate, terminal_growth, forecast_years)
    
    # Monte Carlo Simulation
    with st.spinner("Ejecutando simulación Monte Carlo..."):
        close_prices = prices['Close']
        mc_results = run_monte_carlo(close_prices, n_simulations, mc_days)
    
    fair_value_data = valuations['fair_value']
    
    # Inicializar chatbot en session state si no existe
    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = ValuationChatbot()
        st.session_state['chatbot'].set_context(
            ticker=ticker,
            current_price=current_price,
            fair_value=fair_value_data['fair_value_estimate'],
            valuations=valuations,
            mc_results=mc_results,
            sec_report=st.session_state.get('sec_report', {}),
            summary=summary
        )
    
    # Contenido principal - Tabs principales
    main_tab1, main_tab2, main_tab3 = st.tabs([
        "📊 Valoración Tradicional",
        "🤖 Análisis IA - Reportes SEC",
        "💬 Chatbot Financiero"
    ])
    
    with main_tab2:
        # Nueva sección de análisis con IA
        st.header("🤖 Análisis Inteligente de Reportes 10-K/10-Q")
        
        st.markdown("""
        Esta sección utiliza IA de Google Gemini para analizar automáticamente los reportes 
        financieros 10-K y 10-Q de Ally Financial, extrayendo:
        - **Riesgos clave** identificados por la empresa
        - **Top 10 KPIs** más relevantes
        - **Sentimiento** de la Management Discussion & Analysis (MD&A)
        - **Drivers de ingresos** mencionados
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            filing_type = st.selectbox(
                "Tipo de Reporte",
                ["10-K", "10-Q"],
                help="10-K: Reporte anual completo | 10-Q: Reporte trimestral"
            )
        
        with col2:
            analyze_button = st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True)
        
        if analyze_button:
            try:
                with st.spinner(f"🔍 Analizando {filing_type} de {ticker} con IA..."):
                    # Inicializar analizador
                    analyzer = SECAnalyzer()
                    
                    # Generar reporte completo
                    report = analyzer.generate_full_report(ticker=ticker, filing_type=filing_type)
                    
                    # Mostrar reporte
                    if "error" in report:
                        st.error(f"❌ {report['error']}")
                    else:
                        # Guardar en session state para persistencia
                        st.session_state['sec_report'] = report
                        st.session_state['report_ticker'] = ticker
                        st.session_state['report_type'] = filing_type
                        
                        st.success(f"✅ Análisis completado para {ticker} {filing_type}")
            
            except Exception as e:
                st.error(f"❌ Error durante el análisis: {str(e)}")
                st.info("Verifica que la API de Gemini esté configurada correctamente en el archivo .env")
        
        # Mostrar reporte si existe en session state
        if 'sec_report' in st.session_state:
            st.markdown("---")
            
            report = st.session_state['sec_report']
            
            # Mostrar métricas resumen en cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                num_risks = len(report.get('riesgos', []))
                st.metric("🔴 Riesgos", num_risks)
            
            with col2:
                num_kpis = len(report.get('kpis', []))
                st.metric("📊 KPIs", num_kpis)
            
            with col3:
                sentiment = report.get('sentimiento', {})
                sent_type = sentiment.get('sentimiento_general', {}).get('tipo', 'N/A')
                sent_emoji = {"Positivo": "😊", "Neutral": "😐", "Negativo": "😟"}.get(sent_type, "❓")
                st.metric("💭 Sentimiento", f"{sent_emoji} {sent_type}")
            
            with col4:
                num_drivers = len(report.get('revenue_drivers', []))
                st.metric("💰 Drivers", num_drivers)
            
            st.markdown("---")
            
            # Crear tabs para diferentes secciones del reporte
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Resumen",
                "⚠️ Riesgos",
                "📊 KPIs",
                "💭 Sentimiento",
                "💰 Revenue Drivers"
            ])
            
            with tab1:
                st.markdown("## 📋 Resumen Ejecutivo")
                st.markdown(report.get('resumen_ejecutivo', '*No disponible*'))
                
                # Opción de descargar reporte completo
                st.markdown("---")
                formatted_report = format_report_for_display(report)
                st.download_button(
                    label="📥 Descargar Reporte Completo (Markdown)",
                    data=formatted_report,
                    file_name=f"{report['ticker']}_{report['filing_type']}_AI_Analysis.md",
                    mime="text/markdown"
                )
            
            with tab2:
                st.markdown("## ⚠️ Riesgos Clave Identificados")
                risks = report.get('riesgos', [])
                
                if risks:
                    for i, risk in enumerate(risks, 1):
                        with st.expander(f"**{i}. {risk.get('nombre', 'N/A')}** - {risk.get('severidad', 'N/A')}"):
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.write(f"**Categoría:**")
                                st.write(f"**Severidad:**")
                            with col_b:
                                st.write(risk.get('categoria', 'N/A'))
                                severity = risk.get('severidad', 'N/A')
                                color = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}.get(severity, "⚪")
                                st.write(f"{color} {severity}")
                            
                            st.markdown("**Descripción:**")
                            st.write(risk.get('descripcion', 'N/A'))
                else:
                    st.info("No se identificaron riesgos en el análisis")
            
            with tab3:
                st.markdown("## 📊 Top 10 KPIs")
                kpis = report.get('kpis', [])
                
                if kpis:
                    # Crear DataFrame para mejor visualización
                    kpi_data = []
                    for i, kpi in enumerate(kpis, 1):
                        trend = kpi.get('tendencia', 'N/A')
                        trend_emoji = {
                            "Mejorando": "📈",
                            "Estable": "➡️",
                            "Deteriorando": "📉"
                        }.get(trend, "❓")
                        
                        importance = kpi.get('importancia', 'N/A')
                        imp_emoji = {
                            "Alta": "🔴",
                            "Media": "🟡",
                            "Baja": "🟢"
                        }.get(importance, "⚪")
                        
                        kpi_data.append({
                            "#": i,
                            "KPI": kpi.get('nombre', 'N/A'),
                            "Valor": kpi.get('valor', 'N/A'),
                            "Tendencia": f"{trend_emoji} {trend}",
                            "Importancia": f"{imp_emoji} {importance}"
                        })
                    
                    kpi_df = pd.DataFrame(kpi_data)
                    st.dataframe(kpi_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No se identificaron KPIs en el análisis")
            
            with tab4:
                st.markdown("## 💭 Análisis de Sentimiento (MD&A)")
                sentiment = report.get('sentimiento', {})
                
                if sentiment:
                    # Sentimiento general
                    sent_general = sentiment.get('sentimiento_general', {})
                    sent_type = sent_general.get('tipo', 'N/A')
                    sent_pct = sent_general.get('porcentaje', 0)
                    
                    col_sent1, col_sent2 = st.columns([2, 1])
                    
                    with col_sent1:
                        # Gauge chart para sentimiento
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=sent_pct,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"Sentimiento: {sent_type}"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 33], 'color': "#ffcccc"},
                                    {'range': [33, 66], 'color': "#ffffcc"},
                                    {'range': [66, 100], 'color': "#ccffcc"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': sent_pct
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col_sent2:
                        st.markdown("### Nivel de Confianza")
                        confidence = sentiment.get('nivel_confianza', 'N/A')
                        conf_emoji = {
                            "Alto": "🟢",
                            "Medio": "🟡",
                            "Bajo": "🔴"
                        }.get(confidence, "⚪")
                        st.markdown(f"## {conf_emoji} {confidence}")
                    
                    st.markdown("---")
                    
                    # Temas positivos y preocupaciones
                    col_pos, col_neg = st.columns(2)
                    
                    with col_pos:
                        st.markdown("### 🟢 Temas Positivos")
                        temas_pos = sentiment.get('temas_positivos', [])
                        if temas_pos:
                            for tema in temas_pos:
                                st.markdown(f"- {tema}")
                        else:
                            st.info("No identificados")
                    
                    with col_neg:
                        st.markdown("### 🔴 Preocupaciones")
                        preocupaciones = sentiment.get('preocupaciones', [])
                        if preocupaciones:
                            for preoc in preocupaciones:
                                st.markdown(f"- {preoc}")
                        else:
                            st.info("No identificadas")
                else:
                    st.info("Análisis de sentimiento no disponible")
            
            with tab5:
                st.markdown("## 💰 Drivers de Ingresos")
                drivers = report.get('revenue_drivers', [])
                
                if drivers:
                    for i, driver in enumerate(drivers, 1):
                        with st.expander(f"**{i}. {driver.get('nombre', 'N/A')}**"):
                            col_d1, col_d2, col_d3 = st.columns(3)
                            
                            with col_d1:
                                impact = driver.get('impacto', 'N/A')
                                impact_emoji = {
                                    "Alto": "🔴",
                                    "Medio": "🟡",
                                    "Bajo": "🟢"
                                }.get(impact, "⚪")
                                st.metric("Impacto", f"{impact_emoji} {impact}")
                            
                            with col_d2:
                                trend = driver.get('tendencia', 'N/A')
                                trend_emoji = {
                                    "Creciendo": "📈",
                                    "Estable": "➡️",
                                    "Declinando": "📉"
                                }.get(trend, "❓")
                                st.metric("Tendencia", f"{trend_emoji} {trend}")
                            
                            with col_d3:
                                st.write("")  # Espaciador
                            
                            st.markdown("**Descripción:**")
                            st.write(driver.get('descripcion', 'N/A'))
                else:
                    st.info("No se identificaron drivers de ingresos en el análisis")
    
    with main_tab3:
        # Chatbot Financiero
        st.header("💬 Chatbot Financiero con IA")
        
        st.markdown("""
        Chatea con nuestro asistente financiero inteligente para interpretar los resultados del análisis.
        El chatbot tiene acceso completo a:
        - 📊 Resultados de valoración
        - 📈 Análisis Monte Carlo
        - 🤖 Reportes SEC analizados
        - 💰 Métricas financieras
        """)
        
        # Botones de acción
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🔄 Nueva Conversación"):
                st.session_state['chatbot'].clear_history()
                st.session_state['chatbot'].set_context(
                    ticker=ticker,
                    current_price=current_price,
                    fair_value=fair_value_data['fair_value_estimate'],
                    valuations=valuations,
                    mc_results=mc_results,
                    sec_report=st.session_state.get('sec_report', {}),
                    summary=summary
                )
                st.rerun()
        
        with col_btn2:
            show_suggestions = st.button("💡 Sugerencias")
        
        # Mostrar sugerencias de preguntas
        if show_suggestions or 'show_suggestions' not in st.session_state:
            st.session_state['show_suggestions'] = True
        
        if st.session_state.get('show_suggestions', False):
            st.markdown("### 💡 Preguntas Sugeridas:")
            suggestions = st.session_state['chatbot'].suggest_questions()
            
            cols = st.columns(2)
            for idx, suggestion in enumerate(suggestions):
                col_idx = idx % 2
                with cols[col_idx]:
                    if st.button(f"❓ {suggestion}", key=f"sug_{idx}"):
                        # Agregar pregunta al historial y obtener respuesta
                        response = st.session_state['chatbot'].chat(suggestion)
                        st.session_state['show_suggestions'] = False
                        st.rerun()
        
        st.markdown("---")
        
        # Área de chat
        st.markdown("### 💬 Conversación")
        
        # Contenedor de chat con altura fija
        chat_container = st.container()
        
        with chat_container:
            # Mostrar historial de conversación
            history = st.session_state['chatbot'].get_conversation_history()
            
            if not history:
                st.info("👋 ¡Hola! Soy tu asistente financiero. Pregúntame lo que quieras sobre el análisis de valoración.")
            else:
                for entry in history:
                    # Mensaje del usuario
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 12px; border-radius: 10px; 
                                margin: 8px 0; border-left: 4px solid #2196F3;">
                        <strong>🧑 Tú:</strong><br>{entry['user']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Respuesta del asistente
                    st.markdown(f"""
                    <div style="background-color: #f5f5f5; padding: 12px; border-radius: 10px; 
                                margin: 8px 0; border-left: 4px solid #4CAF50;">
                        <strong>🤖 Asistente:</strong><br>{entry['assistant']}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Input para nueva pregunta
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_submit = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Tu pregunta:",
                    placeholder="Ej: ¿Es buen momento para comprar esta acción?",
                    label_visibility="collapsed"
                )
            
            with col_submit:
                submit_button = st.form_submit_button("📤 Enviar", use_container_width=True)
            
            if submit_button and user_input:
                # Ocultar sugerencias
                st.session_state['show_suggestions'] = False
                
                # Obtener respuesta del chatbot
                with st.spinner("🤔 Pensando..."):
                    response = st.session_state['chatbot'].chat(user_input)
                
                # Recargar para mostrar la nueva conversación
                st.rerun()
        
        # Información adicional
        st.markdown("---")
        with st.expander("ℹ️ Información sobre el Chatbot"):
            st.markdown("""
            **Capacidades:**
            - ✅ Interpreta resultados de valoración
            - ✅ Explica conceptos financieros
            - ✅ Analiza riesgos y oportunidades
            - ✅ Proporciona recomendaciones basadas en datos
            - ✅ Responde preguntas sobre KPIs y métricas
            
            **Limitaciones:**
            - ❌ No tiene acceso a datos en tiempo real fuera del análisis
            - ❌ No es asesoría financiera profesional certificada
            - ❌ Las respuestas son para fines educativos e informativos
            
            **Consejos:**
            - Sé específico en tus preguntas
            - Pregunta sobre aspectos concretos del análisis
            - Usa las preguntas sugeridas como punto de partida
            """)
    
    with main_tab1:
        # Métricas clave
        st.header("📈 Métricas Financieras Clave")
    
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = summary.get('current_price', 0)
        
        with col1:
            st.metric("Precio Actual", f"${current_price:.2f}")
        with col2:
            market_cap = summary.get('market_cap', 0)
            st.metric("Capitalización", f"${market_cap/1e9:.2f}B")
        with col3:
            eps = summary.get('eps', 0)
            st.metric("EPS", f"${eps:.2f}")
        with col4:
            div_yield = summary.get('dividend_yield', 0) * 100
            st.metric("Rendimiento por dividendo", f"{div_yield:.2f}%")
        with col5:
            book_val = summary.get('book_value_per_share', 0)
            st.metric("Valor contable/acción", f"${book_val:.2f}")
        
        # Fila 2: Gráfico de precio
        st.header("📊 Datos Históricos de Precio")
        
        # Create interactive price chart
        fig_price = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price', 'Volume')
        )
        
        fig_price.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices['Open'],
                high=prices['High'],
                low=prices['Low'],
                close=prices['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        fig_price.add_trace(
            go.Bar(x=prices.index, y=prices['Volume'], name='Volume', marker_color='rgba(31, 119, 180, 0.5)'),
            row=2, col=1
        )
        
        fig_price.update_layout(
            height=600,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            title_text=f"{ticker} - Historial de precios (5 años)"
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Run valuations
        st.header("💰 Análisis de Valoración")
        
        with st.spinner("Ejecutando modelos de valoración..."):
            valuations = run_valuations(ticker, required_return, growth_rate, terminal_growth, forecast_years)
        
        # Create tabs for different valuation methods
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📗 Valor Contable", "📘 Valor Contable Ajustado", "📈 Relación P/E", 
            "💵 DDM", "🏢 Empresas Comparables", "💎 DCF/FCF"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Análisis: Valor Contable")
                bv = valuations['book_value']
                st.metric("Valor Intrínseco", f"${bv['intrinsic_value']:.2f}")
                st.metric("Precio / Valor Contable", f"{bv['price_to_book']:.2f}x")
                st.metric("Potencial Alza", f"{bv['upside_potential']:.2f}%", 
                         delta=f"{bv['upside_potential']:.1f}%")
            with col2:
                st.write("**Descripción del método:**")
                st.write("Valor Contable = Patrimonio Total / Acciones en Circulación")
                st.write(f"- Patrimonio Total: ${bv.get('total_equity', 0):,.0f}")
                st.write(f"- Acciones en Circulación: {bv.get('shares_outstanding', 0):,.0f}")
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Análisis: Valor Contable Ajustado")
                abv = valuations['adj_book']
                st.metric("Valor Intrínseco", f"${abv['intrinsic_value']:.2f}")
                st.metric("Precio / Valor Ajustado", f"{abv['price_to_adjusted_book']:.2f}x")
                st.metric("Potencial Alza", f"{abv['upside_potential']:.2f}%",
                         delta=f"{abv['upside_potential']:.1f}%")
            with col2:
                st.write("**Descripción del método:**")
                st.write("Valor Contable Ajustado = (Patrimonio Total - Intangibles) / Acciones")
                st.write(f"- Patrimonio Total: ${abv.get('total_equity', 0):,.0f}")
                st.write(f"- Activos intangibles: ${abv.get('intangible_assets', 0):,.0f}")
                st.write(f"- Patrimonio ajustado: ${abv.get('adjusted_equity', 0):,.0f}")
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Análisis: Relación P/E")
                pe = valuations['pe_val']
                st.metric("Valor Intrínseco", f"${pe['intrinsic_value']:.2f}")
                st.metric("P/E Actual", f"{pe['current_pe']:.2f}x")
                st.metric("P/E Objetivo (Sector)", f"{pe['target_pe']:.2f}x")
                st.metric("Potencial Alza", f"{pe['upside_potential']:.2f}%",
                         delta=f"{pe['upside_potential']:.1f}%")
            with col2:
                st.write("**Descripción del método:**")
                st.write("Valor Intrínseco = EPS × P/E Objetivo")
                st.write(f"- EPS: ${pe.get('eps', 0):.2f}")
                st.write(f"- P/E objetivo: {pe.get('target_pe', 0):.2f}x (mediana sectorial)")
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Modelo de Descuento de Dividendos (DDM)")
                ddm = valuations['ddm']
                if ddm.get('error'):
                    st.warning(ddm['error'])
                else:
                    st.metric("Valor Intrínseco", f"${ddm['intrinsic_value']:.2f}")
                    st.metric("Dividendo/Acción", f"${ddm['dividend_per_share']:.2f}")
                    st.metric("Potencial Alza", f"{ddm['upside_potential']:.2f}%",
                             delta=f"{ddm['upside_potential']:.1f}%")
            with col2:
                st.write("**Parámetros del modelo:**")
                st.write(f"- Rentabilidad requerida: {ddm.get('required_return', 0)*100:.1f}%")
                st.write(f"- Tasa de crecimiento (Etapa 1): {ddm.get('growth_rate_stage1', 0)*100:.1f}%")
                st.write(f"- Crecimiento terminal: {ddm.get('terminal_growth_rate', 0)*100:.1f}%")
                st.write(f"- Años de pronóstico: {ddm.get('forecast_years', 0)}")
                
                if ddm.get('projected_dividends'):
                    st.write("**Dividendos proyectados:**")
                    div_df = pd.DataFrame(ddm['projected_dividends'])
                    st.dataframe(div_df.round(2))
        
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Empresas Comparables")
                comp = valuations['comparables']
                if comp.get('error'):
                    st.warning(comp['error'])
                else:
                    st.metric("Valor Intrínseco", f"${comp['intrinsic_value']:.2f}")
                    st.metric("Valor implícito (P/E)", f"${comp.get('implied_value_pe', 0):.2f}")
                    st.metric("Valor implícito (P/B)", f"${comp.get('implied_value_pb', 0):.2f}")
                    st.metric("Potencial Alza", f"{comp['upside_potential']:.2f}%",
                             delta=f"{comp['upside_potential']:.1f}%")
            with col2:
                st.write("**Métricas de empresas pares:**")
                if comp.get('peer_metrics'):
                    peer_df = pd.DataFrame(comp['peer_metrics'])
                    st.dataframe(peer_df)
                    st.write(f"Mediana P/E: {comp.get('median_pe', 'N/A')}")
                    st.write(f"Mediana P/B: {comp.get('median_pb', 'N/A')}")
        
        with tab6:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("DCF (Flujo de Caja Libre)")
                dcf = valuations['dcf']
                if dcf.get('error'):
                    st.warning(dcf['error'])
                else:
                    st.metric("Valor Intrínseco", f"${dcf['intrinsic_value']:.2f}")
                    st.metric("Flujo de Caja Libre", f"${dcf['free_cash_flow']:,.0f}")
                    st.metric("Valor Empresa", f"${dcf['enterprise_value']:,.0f}")
                    st.metric("Potencial Alza", f"{dcf['upside_potential']:.2f}%",
                             delta=f"{dcf['upside_potential']:.1f}%")
            with col2:
                st.write("**Parámetros del modelo:**")
                st.write(f"- WACC: {dcf.get('wacc', 0)*100:.1f}%")
                st.write(f"- Tasa de crecimiento: {dcf.get('growth_rate', 0)*100:.1f}%")
                st.write(f"- Crecimiento terminal: {dcf.get('terminal_growth_rate', 0)*100:.1f}%")
                
                if dcf.get('projected_fcfs'):
                    st.write("**Flujos de caja proyectados:**")
                    fcf_df = pd.DataFrame(dcf['projected_fcfs'])
                    fcf_df['fcf'] = fcf_df['fcf'].apply(lambda x: f"${x:,.0f}")
                    fcf_df['pv'] = fcf_df['pv'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(fcf_df)
        
        # Valuation Comparison Chart
        st.header("📊 Valuation Methods Comparison")
        
        summary_df = valuations['summary']
        valid_vals = summary_df[summary_df['Intrinsic Value ($)'] > 0].copy()
        
        # Create comparison bar chart
        fig_comparison = go.Figure()
        
        colors = ['#28a745' if v > current_price else '#dc3545' for v in valid_vals['Intrinsic Value ($)']]
        
        fig_comparison.add_trace(go.Bar(
            x=valid_vals['Method'],
            y=valid_vals['Intrinsic Value ($)'],
            marker_color=colors,
            text=[f"${v:.2f}" for v in valid_vals['Intrinsic Value ($)']],
            textposition='outside',
            name='Valor Intrínseco'
        ))
        
        fig_comparison.add_hline(
            y=current_price, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Precio actual: ${current_price:.2f}",
            annotation_position="top right"
        )
        
        fair_val = valuations['fair_value']['fair_value_estimate']
        fig_comparison.add_hline(
            y=fair_val, 
            line_dash="dot", 
            line_color="purple",
            annotation_text=f"Estimación Valor Justo: ${fair_val:.2f}",
            annotation_position="bottom right"
        )
        
        fig_comparison.update_layout(
            title="Valor Intrínseco por Método de Valoración",
            xaxis_title="Método de Valoración",
            yaxis_title="Precio ($)",
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Fair Value Summary
        col1, col2, col3 = st.columns(3)
        
        fair_value_data = valuations['fair_value']
        upside = fair_value_data['upside_potential']
        
        with col1:
            st.metric("📊 Valor Justo Ponderado", f"${fair_value_data['fair_value_estimate']:.2f}")
        with col2:
            st.metric("💵 Precio actual", f"${fair_value_data['current_price']:.2f}")
        with col3:
            delta_color = "normal" if upside > 0 else "inverse"
            st.metric("📈 Upside Potential", f"{upside:.2f}%", delta=f"{upside:.1f}%", delta_color=delta_color)
        
        # Monte Carlo Simulation
        st.header("🎲 Monte Carlo Simulation")
        
        with st.spinner("Running Monte Carlo simulation..."):
            close_prices = prices['Close']
            mc_results = run_monte_carlo(close_prices, n_simulations, mc_days)
        
        # Monte Carlo Results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Final Price", f"${mc_results['mean_final_price']:.2f}")
        with col2:
            st.metric("Probability of Profit", f"{mc_results['prob_profit']:.1f}%")
        with col3:
            st.metric("95% VaR", f"${mc_results['var_95']:.2f}")
        with col4:
            st.metric("Expected Return", f"{mc_results['expected_return_pct']:.2f}%")
        
        # Monte Carlo Charts
        mc_col1, mc_col2 = st.columns(2)
        
        with mc_col1:
            # Price paths
            fig_paths = go.Figure()
            
            # Plot sample paths
            n_sample_paths = 100
            for i in range(min(n_sample_paths, mc_results['price_paths'].shape[0])):
                fig_paths.add_trace(go.Scatter(
                    y=mc_results['price_paths'][i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(31, 119, 180, 0.1)'),
                    showlegend=False
                ))
            
            # Add mean path
            mean_path = np.mean(mc_results['price_paths'], axis=0)
            fig_paths.add_trace(go.Scatter(
                y=mean_path,
                mode='lines',
                line=dict(width=2, color='red'),
                name='Mean Path'
            ))
            
            fig_paths.update_layout(
                title=f"Trayectorias Monte Carlo ({n_simulations:,} simulaciones)",
                xaxis_title="Días de negociación",
                yaxis_title="Precio ($)",
                height=400
            )
            
            st.plotly_chart(fig_paths, use_container_width=True)
        
        with mc_col2:
            # Final price distribution
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=mc_results['final_prices'],
                nbinsx=50,
                name='Final Prices',
                marker_color='rgba(31, 119, 180, 0.7)'
            ))
            
            fig_dist.add_vline(x=mc_results['initial_price'], line_dash="dash", line_color="green",
                      annotation_text=f"Inicial: ${mc_results['initial_price']:.2f}")
            fig_dist.add_vline(x=mc_results['mean_final_price'], line_dash="solid", line_color="red",
                      annotation_text=f"Media: ${mc_results['mean_final_price']:.2f}")
            
            fig_dist.update_layout(
                title="Distribución de Precios Finales",
                xaxis_title="Precio ($)",
                yaxis_title="Frecuencia",
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Monte Carlo Statistics Table
        st.subheader("📊 Monte Carlo Statistics")
        
        mc_stats = {
            "Métrica": [
                "Precio inicial", "Precio medio final", "Precio mediano final", "Desv. estándar",
                "Percentil 5", "Percentil 25", "Percentil 75", "Percentil 95",
                "VaR (95%)", "CVaR (95%)", "Volatilidad anual"
            ],
            "Valor": [
                f"${mc_results['initial_price']:.2f}",
                f"${mc_results['mean_final_price']:.2f}",
                f"${mc_results['median_final_price']:.2f}",
                f"${mc_results['std_final_price']:.2f}",
                f"${mc_results['percentiles'][5]:.2f}",
                f"${mc_results['percentiles'][25]:.2f}",
                f"${mc_results['percentiles'][75]:.2f}",
                f"${mc_results['percentiles'][95]:.2f}",
                f"${mc_results['var_95']:.2f}",
                f"${mc_results['cvar_95']:.2f}",
                f"{mc_results['volatility_annual']*100:.2f}%"
            ]
        }
        
        st.dataframe(pd.DataFrame(mc_stats), use_container_width=True)
        
        # Sección LSTM (Opcional)
        st.header("🤖 Predicción con LSTM")
        
        run_lstm = st.checkbox("Ejecutar modelo LSTM (puede tardar varios minutos)")
        
        if run_lstm:
            try:
                from src.lstm_model import LSTMPredictor
                
                with st.spinner("Entrenando modelo LSTM... Esto puede tardar varios minutos."):
                    lstm = LSTMPredictor(
                        sequence_length=60,
                        units=50,
                        epochs=25,
                        batch_size=32
                    )
                    
                    training_results = lstm.train(close_prices, train_ratio=0.8, verbose=0)
                    
                    # Predict future
                    days_ahead = 30
                    future_predictions = lstm.predict_future(close_prices, days_ahead=days_ahead)
                    
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RMSE entrenamiento", f"${training_results['train_rmse']:.2f}")
                with col2:
                    st.metric("RMSE prueba", f"${training_results['test_rmse']:.2f}")
                with col3:
                    predicted_change = ((future_predictions[-1] - close_prices.iloc[-1]) / close_prices.iloc[-1] * 100)
                    st.metric(f"Predicted Price ({days_ahead}d)", f"${future_predictions[-1]:.2f}",
                             delta=f"{predicted_change:.2f}%")
                
                # LSTM Prediction Chart
                fig_lstm = go.Figure()
                
                # Historical data (last 90 days)
                hist_data = close_prices.iloc[-90:]
                fig_lstm.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Future predictions
                future_dates = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), 
                                            periods=days_ahead, freq='B')
                fig_lstm.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines',
                    name='LSTM Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_lstm.update_layout(
                    title=f"Pronóstico LSTM a {days_ahead} días",
                    xaxis_title="Fecha",
                    yaxis_title="Precio ($)",
                    height=400
                )
                
                st.plotly_chart(fig_lstm, use_container_width=True)
                
            except ImportError:
                st.warning("TensorFlow es necesario para LSTM. Instálalo con: pip install tensorflow")
            except Exception as e:
                st.error(f"Error running LSTM model: {str(e)}")
        
        # Recomendación final
        st.header("📋 Recomendación de inversión")
        
        upside = fair_value_data['upside_potential']
        
        if upside > 20:
            recommendation = "COMPRA FIRME"
            rec_color = "recommendation-buy"
            description = "La acción parece significativamente infravalorada según múltiples métodos."
        elif upside > 10:
            recommendation = "COMPRAR"
            rec_color = "recommendation-buy"
            description = "La acción parece moderadamente infravalorada y puede ofrecer retornos atractivos."
        elif upside > -10:
            recommendation = "MANTENER"
            rec_color = "recommendation-hold"
            description = "La acción parece razonablemente valorada en los niveles actuales."
        elif upside > -20:
            recommendation = "VENDER"
            rec_color = "recommendation-sell"
            description = "La acción parece moderadamente sobrevalorada en los precios actuales."
        else:
            recommendation = "VENTA FUERTE"
            rec_color = "recommendation-sell"
            description = "La acción parece significativamente sobrevalorada según múltiples métodos."
        
        st.markdown(f"""
        <div class="metric-card">
            <h2 class="{rec_color}">{recommendation}</h2>
            <p>{description}</p>
            <ul>
                <li>Precio actual: ${current_price:.2f}</li>
                <li>Fair Value Estimate: ${fair_value_data['fair_value_estimate']:.2f}</li>
                <li>Upside/Downside Potential: {upside:.2f}%</li>
                <li>Probabilidad de Ganancia (Monte Carlo, 1 año): {mc_results['prob_profit']:.1f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your internet connection and ensure the ticker symbol is valid.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Desarrollado con Streamlit | Datos de Yahoo Finance | © 2025 | Grupo 8 </p>
</div>
""", unsafe_allow_html=True)
