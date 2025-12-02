"""
Chatbot inteligente con Gemini AI para interpretar resultados de valoración.
"""

import os
import json
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class ValuationChatbot:
    """Chatbot inteligente para interpretar resultados de valoración."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar el chatbot.
        
        Args:
            api_key: API key de Gemini (opcional, se puede cargar del .env)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada. Asegúrate de tenerla en el archivo .env")
        
        genai.configure(api_key=self.api_key)
        # Usar gemini-pro que es más estable
        self.model = genai.GenerativeModel('gemini-pro')
        self.chat_session = None
        self.context = {}
        self.conversation_history = []
    
    def set_context(self, 
                    ticker: str,
                    current_price: float,
                    fair_value: float,
                    valuations: Dict = None,
                    mc_results: Dict = None,
                    sec_report: Dict = None,
                    summary: Dict = None,
                    revenue_summary: Dict = None):
        """
        Establecer el contexto del análisis para el chatbot.
        
        Args:
            ticker: Símbolo de la empresa
            current_price: Precio actual de la acción
            fair_value: Valor justo estimado
            valuations: Resultados de valoración
            mc_results: Resultados de Monte Carlo
            sec_report: Reporte de análisis SEC
            summary: Resumen financiero
            revenue_summary: Análisis de revenue
        """
        self.context = {
            'ticker': ticker,
            'current_price': current_price,
            'fair_value': fair_value,
            'upside_potential': ((fair_value - current_price) / current_price * 100) if current_price > 0 else 0,
            'valuations': valuations or {},
            'mc_results': mc_results or {},
            'sec_report': sec_report or {},
            'summary': summary or {},
            'revenue_summary': revenue_summary or {}
        }
        
        # Iniciar nueva sesión de chat con contexto
        self._initialize_chat_session()
    
    def _initialize_chat_session(self):
        """Inicializar sesión de chat con contexto del análisis."""
        context_prompt = self._build_context_prompt()
        
        self.chat_session = self.model.start_chat(history=[])
        
        # Enviar contexto inicial (no se mostrará al usuario)
        initial_message = f"""
        Eres un asistente financiero experto especializado en análisis de valoración de acciones.
        Tienes acceso al siguiente contexto del análisis de {self.context.get('ticker', 'N/A')}:
        
        {context_prompt}
        
        Tu trabajo es:
        1. Interpretar los resultados del análisis de manera clara y profesional
        2. Responder preguntas sobre la valoración, riesgos, y proyecciones
        3. Proporcionar insights y recomendaciones basadas en los datos
        4. Explicar conceptos financieros de manera accesible
        5. Ser conciso pero informativo (máximo 3-4 párrafos por respuesta)
        
        Responde SOLO a preguntas relacionadas con el análisis financiero y la empresa.
        Si te preguntan algo no relacionado, indica educadamente que solo puedes ayudar con temas financieros.
        
        Usa emojis relevantes para hacer las respuestas más visuales.
        Responde en español.
        """
        
        self.chat_session.send_message(initial_message)
    
    def _build_context_prompt(self) -> str:
        """Construir prompt de contexto con los datos disponibles."""
        parts = []
        
        # Información básica
        parts.append(f"""
        **Información Básica:**
        - Ticker: {self.context.get('ticker', 'N/A')}
        - Precio Actual: ${self.context.get('current_price', 0):.2f}
        - Valor Justo Estimado: ${self.context.get('fair_value', 0):.2f}
        - Upside Potential: {self.context.get('upside_potential', 0):.2f}%
        """)
        
        # Resumen financiero
        summary = self.context.get('summary', {})
        if summary:
            parts.append(f"""
            **Métricas Financieras:**
            - Market Cap: ${summary.get('market_cap', 0)/1e9:.2f}B
            - EPS: ${summary.get('eps', 0):.2f}
            - P/E Ratio: {summary.get('pe_ratio', 0):.2f}
            - Dividend Yield: {summary.get('dividend_yield', 0)*100:.2f}%
            - Book Value/Share: ${summary.get('book_value_per_share', 0):.2f}
            """)
        
        # Resultados de valoración
        valuations = self.context.get('valuations', {})
        if valuations:
            summary_df = valuations.get('summary')
            if summary_df is not None:
                parts.append(f"""
                **Valoraciones por Método:**
                {summary_df.to_string()}
                """)
        
        # Resultados de Monte Carlo
        mc_results = self.context.get('mc_results', {})
        if mc_results:
            parts.append(f"""
            **Simulación Monte Carlo:**
            - Precio Medio Proyectado: ${mc_results.get('mean_final_price', 0):.2f}
            - Probabilidad de Ganancia: {mc_results.get('prob_profit', 0):.1f}%
            - VaR (95%): ${mc_results.get('var_95', 0):.2f}
            - Retorno Esperado: {mc_results.get('expected_return_pct', 0):.2f}%
            """)
        
        # Reporte SEC
        sec_report = self.context.get('sec_report', {})
        if sec_report:
            num_risks = len(sec_report.get('riesgos', []))
            num_kpis = len(sec_report.get('kpis', []))
            sentiment = sec_report.get('sentimiento', {})
            sent_type = sentiment.get('sentimiento_general', {}).get('tipo', 'N/A')
            
            parts.append(f"""
            **Análisis de Reportes SEC:**
            - Riesgos Identificados: {num_risks}
            - KPIs Analizados: {num_kpis}
            - Sentimiento MD&A: {sent_type}
            """)
            
            # Incluir algunos riesgos clave
            if sec_report.get('riesgos'):
                parts.append("\n**Principales Riesgos:**")
                for i, risk in enumerate(sec_report['riesgos'][:3], 1):
                    parts.append(f"{i}. {risk.get('nombre', 'N/A')} ({risk.get('severidad', 'N/A')})")
            
            # Incluir algunos KPIs
            if sec_report.get('kpis'):
                parts.append("\n**KPIs Clave:**")
                for i, kpi in enumerate(sec_report['kpis'][:5], 1):
                    parts.append(f"{i}. {kpi.get('nombre', 'N/A')}: {kpi.get('valor', 'N/A')}")
        
        # Análisis de Revenue
        revenue_summary = self.context.get('revenue_summary', {})
        if revenue_summary:
            ttm_rev = revenue_summary.get('ttm_revenue', 0)
            latest_q = revenue_summary.get('latest_quarterly_revenue', 0)
            growth_metrics = revenue_summary.get('growth_metrics', {})
            
            parts.append(f"""
            **Análisis de Revenue:**
            - Revenue TTM: ${ttm_rev/1e9:.2f}B
            - Último Trimestre: ${latest_q/1e9:.2f}B
            - Crecimiento YoY: {growth_metrics.get('latest_growth', 0):.2f}%
            - CAGR: {growth_metrics.get('cagr', 0):.2f}%
            - Crecimiento Promedio: {growth_metrics.get('avg_annual_growth', 0):.2f}%
            """)
            
            # Drivers de revenue
            drivers_data = revenue_summary.get('revenue_drivers', {})
            if drivers_data.get('success') and drivers_data.get('drivers'):
                parts.append("\n**Principales Drivers de Revenue:**")
                for i, driver in enumerate(drivers_data['drivers'][:3], 1):
                    parts.append(f"{i}. {driver.get('driver', 'N/A')} (Importancia: {driver.get('importance', 'N/A')})")
        
        return "\n".join(parts)
    
    def chat(self, user_message: str) -> str:
        """
        Enviar mensaje al chatbot y obtener respuesta.
        
        Args:
            user_message: Mensaje del usuario
            
        Returns:
            Respuesta del chatbot
        """
        if not self.chat_session:
            return "⚠️ El chatbot no ha sido inicializado con contexto. Por favor, carga primero los datos del análisis."
        
        try:
            # Enviar mensaje
            response = self.chat_session.send_message(user_message)
            
            # Guardar en historial
            self.conversation_history.append({
                'user': user_message,
                'assistant': response.text
            })
            
            return response.text
        
        except Exception as e:
            return f"❌ Error al procesar mensaje: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Obtener historial de conversación."""
        return self.conversation_history
    
    def clear_history(self):
        """Limpiar historial de conversación."""
        self.conversation_history = []
        if self.context:
            self._initialize_chat_session()
    
    def suggest_questions(self) -> List[str]:
        """Sugerir preguntas relevantes basadas en el contexto."""
        suggestions = [
            "¿Es buen momento para comprar esta acción?",
            "¿Cuáles son los principales riesgos de invertir?",
            "Explica el análisis de valoración de manera simple",
            "¿Qué dicen los KPIs sobre la salud financiera?",
            "¿Cómo interpreto el análisis de Monte Carlo?",
            "¿Qué método de valoración es más confiable?",
            "Resume el sentimiento del management",
            "¿Cuál es el potencial de crecimiento?"
        ]
        
        # Personalizar basado en contexto
        upside = self.context.get('upside_potential', 0)
        
        if upside > 20:
            suggestions.insert(0, "¿Por qué está tan infravalorada la acción?")
        elif upside < -10:
            suggestions.insert(0, "¿Por qué está sobrevalorada la acción?")
        
        # Agregar preguntas específicas si hay reporte SEC
        if self.context.get('sec_report'):
            suggestions.append("¿Cuáles son los drivers de ingresos principales?")
            suggestions.append("Analiza los riesgos identificados en el 10-K")
        
        return suggestions[:8]  # Retornar máximo 8 sugerencias


def format_chat_message(message: str, is_user: bool = False) -> str:
    """
    Formatear mensaje de chat para visualización en Streamlit.
    
    Args:
        message: Texto del mensaje
        is_user: True si es mensaje del usuario, False si es del asistente
        
    Returns:
        HTML formateado del mensaje
    """
    if is_user:
        return f"""
        <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; 
                    margin: 5px 0; text-align: right;">
            <strong>🧑 Tú:</strong> {message}
        </div>
        """
    else:
        return f"""
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; 
                    margin: 5px 0;">
            <strong>🤖 Asistente:</strong> {message}
        </div>
        """
