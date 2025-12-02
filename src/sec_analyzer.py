"""
Módulo para análisis automático de reportes 10-K/10-Q de Ally Financial usando Gemini AI.
"""

import os
import re
import requests
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class SECAnalyzer:
    """Analizador de reportes SEC usando IA de Gemini."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar el analizador.
        
        Args:
            api_key: API key de Gemini (opcional, se puede cargar del .env)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada. Asegúrate de tenerla en el archivo .env")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def fetch_sec_filing(self, ticker: str = "ALLY", filing_type: str = "10-K") -> Optional[str]:
        """
        Obtener el último filing de SEC para el ticker especificado.
        
        Args:
            ticker: Símbolo de la empresa (default: ALLY)
            filing_type: Tipo de filing (10-K o 10-Q)
            
        Returns:
            Contenido del filing o None si hay error
        """
        try:
            # URL de ejemplo - En producción, usarías la API oficial de SEC
            # Por ahora, retornamos texto de ejemplo para demostración
            print(f"📄 Obteniendo último {filing_type} de {ticker}...")
            
            # Nota: Aquí deberías implementar la lógica real para descargar del SEC EDGAR
            # Por ahora, retornamos un texto de ejemplo
            return self._get_sample_filing_text(ticker, filing_type)
            
        except Exception as e:
            print(f"❌ Error al obtener filing: {str(e)}")
            return None
    
    def _get_sample_filing_text(self, ticker: str, filing_type: str) -> str:
        """Texto de ejemplo para demostración."""
        return f"""
        ALLY FINANCIAL INC. - {filing_type}
        
        ITEM 1A. RISK FACTORS
        
        Our business faces various risks including:
        
        1. Credit Risk: We are exposed to credit risk in our consumer automotive finance business. 
        Deterioration in the credit quality of our portfolio could adversely affect our financial condition.
        
        2. Interest Rate Risk: Changes in interest rates affect our net interest margin and the value 
        of our financial instruments. Rising rates could reduce demand for auto loans.
        
        3. Regulatory Risk: We operate in a highly regulated environment. Changes in regulations 
        could increase our compliance costs and limit our business activities.
        
        4. Competition Risk: The automotive finance market is highly competitive. Increased competition 
        could pressure our margins and market share.
        
        5. Economic Risk: Economic downturns affect consumer ability to repay loans and demand for 
        auto financing. Recession could significantly impact our business.
        
        MANAGEMENT'S DISCUSSION AND ANALYSIS (MD&A)
        
        Financial Performance Overview:
        
        Revenue Growth: Total net revenue increased 8% year-over-year driven by:
        - Strong auto loan originations up 12%
        - Improved net interest margin expansion of 25 basis points
        - Growth in insurance premium revenue
        
        Key Performance Indicators:
        - Return on Equity (ROE): 14.2% (up from 12.8%)
        - Net Interest Margin: 3.85% (up from 3.60%)
        - Efficiency Ratio: 42% (improved from 45%)
        - Loan-to-Deposit Ratio: 85%
        - Non-Performing Loan Ratio: 1.2%
        - Tier 1 Capital Ratio: 10.5%
        - Cost-to-Income Ratio: 38%
        - Customer Retention Rate: 87%
        - Digital Channel Adoption: 68%
        - Average Loan Yield: 6.2%
        
        Positive Developments:
        We are pleased with our strong performance this quarter. Our strategic initiatives in digital 
        banking are gaining significant traction. We're confident in our ability to navigate the 
        current environment and deliver sustainable growth. The expansion of our dealer network 
        positions us well for continued success.
        
        Challenges and Outlook:
        While we remain cautiously optimistic, we acknowledge headwinds from potential economic 
        uncertainty. However, our diversified business model and strong capital position provide 
        resilience. We continue to invest in technology and innovation to enhance customer experience.
        
        Strategic Priorities:
        - Expanding our digital capabilities
        - Strengthening our dealer relationships
        - Optimizing our funding mix
        - Enhancing credit risk management
        - Investing in data analytics and AI
        """
    
    def analyze_risks(self, filing_text: str) -> Dict[str, any]:
        """
        Extraer y analizar los riesgos clave del filing.
        
        Args:
            filing_text: Texto del filing
            
        Returns:
            Diccionario con riesgos identificados
        """
        prompt = f"""
        Analiza el siguiente reporte financiero de Ally Financial y extrae los riesgos clave mencionados.
        
        Para cada riesgo, proporciona:
        1. Nombre del riesgo
        2. Descripción breve
        3. Nivel de severidad (Alto, Medio, Bajo)
        4. Categoría (Crédito, Mercado, Operacional, Regulatorio, etc.)
        
        Reporte:
        {filing_text[:3000]}
        
        Responde en formato JSON con esta estructura:
        {{
            "riesgos": [
                {{
                    "nombre": "...",
                    "descripcion": "...",
                    "severidad": "...",
                    "categoria": "..."
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"❌ Error al analizar riesgos: {str(e)}")
            return {"riesgos": []}
    
    def extract_kpis(self, filing_text: str) -> Dict[str, any]:
        """
        Extraer los KPIs más mencionados del filing.
        
        Args:
            filing_text: Texto del filing
            
        Returns:
            Diccionario con KPIs identificados
        """
        prompt = f"""
        Analiza el siguiente reporte financiero y extrae los 10 KPIs (Key Performance Indicators) 
        más importantes mencionados.
        
        Para cada KPI proporciona:
        1. Nombre del KPI
        2. Valor actual (si está disponible)
        3. Tendencia (Mejorando, Estable, Deteriorando)
        4. Importancia (Alta, Media, Baja)
        
        Reporte:
        {filing_text[:3000]}
        
        Responde en formato JSON con esta estructura:
        {{
            "kpis": [
                {{
                    "nombre": "...",
                    "valor": "...",
                    "tendencia": "...",
                    "importancia": "..."
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"❌ Error al extraer KPIs: {str(e)}")
            return {"kpis": []}
    
    def analyze_sentiment(self, filing_text: str) -> Dict[str, any]:
        """
        Analizar el sentimiento de la sección MD&A (Management Discussion & Analysis).
        
        Args:
            filing_text: Texto del filing
            
        Returns:
            Diccionario con análisis de sentimiento
        """
        prompt = f"""
        Analiza el tono y sentimiento de la sección "Management Discussion & Analysis" (MD&A) 
        del siguiente reporte de Ally Financial.
        
        Proporciona:
        1. Sentimiento general (Positivo, Neutral, Negativo) con porcentaje
        2. Temas positivos mencionados (lista de 3-5)
        3. Preocupaciones o desafíos mencionados (lista de 3-5)
        4. Nivel de confianza del management (Alto, Medio, Bajo)
        5. Palabras clave más frecuentes (10 palabras)
        
        Reporte:
        {filing_text[:3000]}
        
        Responde en formato JSON con esta estructura:
        {{
            "sentimiento_general": {{"tipo": "...", "porcentaje": ...}},
            "temas_positivos": [...],
            "preocupaciones": [...],
            "nivel_confianza": "...",
            "palabras_clave": [...]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"❌ Error al analizar sentimiento: {str(e)}")
            return {}
    
    def identify_revenue_drivers(self, filing_text: str) -> Dict[str, any]:
        """
        Identificar los principales drivers de ingresos mencionados.
        
        Args:
            filing_text: Texto del filing
            
        Returns:
            Diccionario con drivers de ingresos
        """
        prompt = f"""
        Identifica los principales drivers de ingresos mencionados en el reporte de Ally Financial.
        
        Para cada driver proporciona:
        1. Nombre del driver
        2. Impacto (Alto, Medio, Bajo)
        3. Tendencia (Creciendo, Estable, Declinando)
        4. Descripción breve
        
        Reporte:
        {filing_text[:3000]}
        
        Responde en formato JSON con esta estructura:
        {{
            "revenue_drivers": [
                {{
                    "nombre": "...",
                    "impacto": "...",
                    "tendencia": "...",
                    "descripcion": "..."
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            print(f"❌ Error al identificar drivers de ingresos: {str(e)}")
            return {"revenue_drivers": []}
    
    def generate_full_report(self, ticker: str = "ALLY", filing_type: str = "10-K") -> Dict[str, any]:
        """
        Generar informe completo del análisis del filing.
        
        Args:
            ticker: Símbolo de la empresa
            filing_type: Tipo de filing (10-K o 10-Q)
            
        Returns:
            Diccionario con el informe completo
        """
        print(f"\n🤖 Iniciando análisis de {filing_type} para {ticker}...\n")
        
        # 1. Obtener el filing
        filing_text = self.fetch_sec_filing(ticker, filing_type)
        if not filing_text:
            return {"error": "No se pudo obtener el filing"}
        
        # 2. Realizar todos los análisis
        print("📊 Analizando riesgos...")
        risks = self.analyze_risks(filing_text)
        
        print("📈 Extrayendo KPIs...")
        kpis = self.extract_kpis(filing_text)
        
        print("💭 Analizando sentimiento...")
        sentiment = self.analyze_sentiment(filing_text)
        
        print("💰 Identificando drivers de ingresos...")
        revenue_drivers = self.identify_revenue_drivers(filing_text)
        
        # 3. Compilar informe completo
        report = {
            "ticker": ticker,
            "filing_type": filing_type,
            "riesgos": risks.get("riesgos", []),
            "kpis": kpis.get("kpis", []),
            "sentimiento": sentiment,
            "revenue_drivers": revenue_drivers.get("revenue_drivers", []),
            "resumen_ejecutivo": self._generate_executive_summary(
                risks, kpis, sentiment, revenue_drivers
            )
        }
        
        print("\n✅ Análisis completado!\n")
        return report
    
    def _generate_executive_summary(self, risks, kpis, sentiment, revenue_drivers) -> str:
        """Generar resumen ejecutivo del análisis."""
        num_risks = len(risks.get("riesgos", []))
        num_kpis = len(kpis.get("kpis", []))
        sentiment_type = sentiment.get("sentimiento_general", {}).get("tipo", "N/A")
        num_drivers = len(revenue_drivers.get("revenue_drivers", []))
        
        summary = f"""
        **Resumen Ejecutivo del Análisis**
        
        - **Riesgos Identificados**: {num_risks} riesgos clave detectados
        - **KPIs Analizados**: {num_kpis} indicadores de desempeño
        - **Sentimiento General**: {sentiment_type}
        - **Drivers de Ingresos**: {num_drivers} drivers principales identificados
        
        El análisis proporciona una visión comprensiva del estado financiero y estratégico 
        de la compañía basado en su último reporte SEC.
        """
        
        return summary.strip()
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parsear respuesta JSON de Gemini."""
        import json
        
        # Limpiar la respuesta (remover markdown si existe)
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        try:
            return json.loads(clean_text.strip())
        except json.JSONDecodeError as e:
            print(f"⚠️ Error al parsear JSON: {str(e)}")
            print(f"Respuesta: {response_text[:200]}...")
            return {}


def format_report_for_display(report: Dict) -> str:
    """
    Formatear el reporte para visualización en Streamlit.
    
    Args:
        report: Diccionario con el reporte generado
        
    Returns:
        String con el reporte formateado en Markdown
    """
    if "error" in report:
        return f"❌ **Error**: {report['error']}"
    
    output = []
    
    # Header
    output.append(f"# 📊 Análisis de {report['filing_type']} - {report['ticker']}")
    output.append("\n---\n")
    
    # Resumen Ejecutivo
    output.append("## 📋 Resumen Ejecutivo")
    output.append(report.get('resumen_ejecutivo', 'No disponible'))
    output.append("\n---\n")
    
    # Riesgos
    output.append("## ⚠️ Riesgos Clave Identificados")
    risks = report.get('riesgos', [])
    if risks:
        for i, risk in enumerate(risks, 1):
            output.append(f"\n### {i}. {risk.get('nombre', 'N/A')}")
            output.append(f"- **Categoría**: {risk.get('categoria', 'N/A')}")
            output.append(f"- **Severidad**: {risk.get('severidad', 'N/A')}")
            output.append(f"- **Descripción**: {risk.get('descripcion', 'N/A')}")
    else:
        output.append("\n*No se identificaron riesgos*")
    
    output.append("\n---\n")
    
    # KPIs
    output.append("## 📈 Top 10 KPIs")
    kpis = report.get('kpis', [])
    if kpis:
        for i, kpi in enumerate(kpis, 1):
            output.append(f"\n### {i}. {kpi.get('nombre', 'N/A')}")
            output.append(f"- **Valor**: {kpi.get('valor', 'N/A')}")
            output.append(f"- **Tendencia**: {kpi.get('tendencia', 'N/A')}")
            output.append(f"- **Importancia**: {kpi.get('importancia', 'N/A')}")
    else:
        output.append("\n*No se identificaron KPIs*")
    
    output.append("\n---\n")
    
    # Sentimiento
    output.append("## 💭 Análisis de Sentimiento (MD&A)")
    sentiment = report.get('sentimiento', {})
    if sentiment:
        sent_general = sentiment.get('sentimiento_general', {})
        output.append(f"\n**Sentimiento General**: {sent_general.get('tipo', 'N/A')} "
                     f"({sent_general.get('porcentaje', 'N/A')}%)")
        output.append(f"\n**Nivel de Confianza del Management**: "
                     f"{sentiment.get('nivel_confianza', 'N/A')}")
        
        output.append("\n### 🟢 Temas Positivos:")
        for tema in sentiment.get('temas_positivos', []):
            output.append(f"- {tema}")
        
        output.append("\n### 🔴 Preocupaciones:")
        for preoc in sentiment.get('preocupaciones', []):
            output.append(f"- {preoc}")
        
        output.append("\n### 🔑 Palabras Clave:")
        output.append(", ".join(sentiment.get('palabras_clave', [])))
    else:
        output.append("\n*No disponible*")
    
    output.append("\n---\n")
    
    # Revenue Drivers
    output.append("## 💰 Drivers de Ingresos")
    drivers = report.get('revenue_drivers', [])
    if drivers:
        for i, driver in enumerate(drivers, 1):
            output.append(f"\n### {i}. {driver.get('nombre', 'N/A')}")
            output.append(f"- **Impacto**: {driver.get('impacto', 'N/A')}")
            output.append(f"- **Tendencia**: {driver.get('tendencia', 'N/A')}")
            output.append(f"- **Descripción**: {driver.get('descripcion', 'N/A')}")
    else:
        output.append("\n*No se identificaron drivers de ingresos*")
    
    return "\n".join(output)
