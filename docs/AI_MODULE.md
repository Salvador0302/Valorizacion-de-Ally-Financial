# 🤖 Módulo de Análisis de Reportes SEC con IA + Chatbot Financiero

## Descripción

Este módulo utiliza Google Gemini AI para:
1. Analizar automáticamente los reportes financieros 10-K y 10-Q de Ally Financial
2. Proporcionar un chatbot inteligente que interpreta los resultados del análisis

## Características

### 📊 Análisis Automatizado

El módulo analiza automáticamente los reportes SEC y extrae:

1. **⚠️ Riesgos Clave**
   - Identificación de riesgos mencionados por la empresa
   - Categorización (Crédito, Mercado, Operacional, Regulatorio, etc.)
   - Nivel de severidad (Alto, Medio, Bajo)
   - Descripción detallada de cada riesgo

2. **📈 Top 10 KPIs**
   - Indicadores de desempeño más relevantes
   - Valores actuales
   - Tendencias (Mejorando, Estable, Deteriorando)
   - Nivel de importancia

3. **💭 Análisis de Sentimiento**
   - Sentimiento general del MD&A (Positivo, Neutral, Negativo)
   - Nivel de confianza del management
   - Temas positivos mencionados
   - Preocupaciones identificadas
   - Palabras clave más frecuentes

4. **💰 Drivers de Ingresos**
   - Principales fuentes de ingresos
   - Impacto de cada driver
   - Tendencias de crecimiento
   - Descripciones detalladas

5. **💬 Chatbot Financiero Inteligente**
   - Interpreta todos los resultados del análisis
   - Responde preguntas sobre valoración, riesgos y proyecciones
   - Proporciona recomendaciones personalizadas
   - Explica conceptos financieros de manera accesible
   - Context-aware: tiene acceso a todos los datos del análisis
   - Sugerencias inteligentes de preguntas relevantes

## Configuración

### 1. API de Google Gemini

Para usar este módulo necesitas una API key de Google Gemini:

1. Visita [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea un nuevo API key
3. Agrega el API key al archivo `.env`:

```bash
GEMINI_API_KEY=tu_api_key_aqui
```

### 2. Dependencias

Las dependencias necesarias ya están incluidas en `requirements.txt`:

```bash
google-generativeai>=0.3.0
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
```

Instala con:

```bash
pip install -r requirements.txt
```

## Uso

### Desde Streamlit

1. Ejecuta la aplicación:
```bash
streamlit run streamlit_app.py
```

2. Navega al tab **"🤖 Análisis IA - Reportes SEC"**

3. Selecciona el tipo de reporte (10-K o 10-Q)

4. Haz clic en **"🚀 Iniciar Análisis"**

5. Espera mientras la IA analiza el reporte (puede tardar 1-2 minutos)

6. Revisa los resultados en las diferentes pestañas:
   - 📋 Resumen
   - ⚠️ Riesgos
   - 📊 KPIs
   - 💭 Sentimiento
   - 💰 Revenue Drivers

### Usando el Chatbot

1. Navega al tab **"💬 Chatbot Financiero"**

2. El chatbot tiene acceso automático a:
   - Todos los resultados de valoración
   - Análisis de Monte Carlo
   - Reportes SEC analizados
   - Métricas financieras

3. **Opciones disponibles:**
   - 💡 **Ver Sugerencias**: Muestra preguntas relevantes predefinidas
   - 🔄 **Nueva Conversación**: Reinicia el chat manteniendo el contexto
   - 📤 **Enviar**: Envía tu pregunta personalizada

4. **Ejemplos de preguntas:**
   - "¿Es buen momento para comprar esta acción?"
   - "Explica el análisis de valoración de manera simple"
   - "¿Cuáles son los principales riesgos?"
   - "¿Qué dicen los KPIs sobre la salud financiera?"
   - "Resume el sentimiento del management"
   - "¿Cómo interpreto la simulación Monte Carlo?"

5. **Tips para mejores respuestas:**
   - Sé específico en tus preguntas
   - Pregunta sobre aspectos concretos del análisis
   - Usa las sugerencias como punto de partida
   - El chatbot mantiene el contexto de la conversación

### Desde Python

```python
from src.sec_analyzer import SECAnalyzer, format_report_for_display

# Inicializar el analizador
analyzer = SECAnalyzer()

# Generar reporte completo
report = analyzer.generate_full_report(ticker="ALLY", filing_type="10-K")

# Mostrar reporte formateado
print(format_report_for_display(report))

# Acceder a secciones específicas
print("Riesgos:", report['riesgos'])
print("KPIs:", report['kpis'])
print("Sentimiento:", report['sentimiento'])
print("Revenue Drivers:", report['revenue_drivers'])

# Usar el Chatbot
from src.chatbot import ValuationChatbot

chatbot = ValuationChatbot()

# Establecer contexto
chatbot.set_context(
    ticker="ALLY",
    current_price=35.50,
    fair_value=42.00,
    valuations=valuations,
    mc_results=mc_results,
    sec_report=report,
    summary=summary
)

# Hacer preguntas
response = chatbot.chat("¿Es buen momento para comprar?")
print(response)

# Obtener sugerencias
suggestions = chatbot.suggest_questions()
print("Preguntas sugeridas:", suggestions)

# Ver historial
history = chatbot.get_conversation_history()
```

## Estructura del Código

### `src/sec_analyzer.py`

Módulo principal que contiene:

- **`SECAnalyzer`**: Clase principal para análisis
  - `fetch_sec_filing()`: Obtiene reportes SEC
  - `analyze_risks()`: Analiza riesgos
  - `extract_kpis()`: Extrae KPIs
  - `analyze_sentiment()`: Analiza sentimiento
  - `identify_revenue_drivers()`: Identifica drivers
  - `generate_full_report()`: Genera reporte completo

- **`format_report_for_display()`**: Formatea reporte para visualización

### `src/chatbot.py`

Módulo del chatbot inteligente:

- **`ValuationChatbot`**: Clase principal del chatbot
  - `set_context()`: Establece contexto del análisis
  - `chat()`: Envía mensaje y obtiene respuesta
  - `suggest_questions()`: Genera preguntas relevantes
  - `get_conversation_history()`: Obtiene historial
  - `clear_history()`: Limpia conversación
  - `_build_context_prompt()`: Construye prompt con datos

- **`format_chat_message()`**: Formatea mensajes para UI

### Integración en Streamlit

El módulo está completamente integrado en `streamlit_app.py` con:

- **Tab de Análisis SEC**: Interface interactiva para reportes
- **Tab de Chatbot**: Conversación inteligente con contexto
- Visualizaciones con gráficos (Plotly)
- Descarga de reportes en formato Markdown
- Sistema de caché para persistencia de datos
- Session state para mantener conversaciones
- Sugerencias inteligentes de preguntas
- Actualización automática de contexto

## Ejemplo de Uso del Chatbot

**Usuario:** "¿Es buen momento para comprar esta acción?"

**Asistente:** "📊 Basándome en el análisis actual de ALLY:

**Indicadores Positivos:**
- El precio actual ($35.50) está 18.3% por debajo del valor justo estimado ($42.00)
- La simulación Monte Carlo muestra 67% de probabilidad de ganancia
- Los KPIs muestran tendencias positivas en ROE y margen de interés neto

**Consideraciones:**
- Los riesgos identificados incluyen exposición a tasas de interés y calidad crediticia
- El sentimiento del management es moderadamente positivo

**Conclusión:** 
Parece ser un momento favorable para considerar la compra, con potencial de upside del 18%. Sin embargo, evalúa tu tolerancia al riesgo y considera los riesgos del sector financiero. 🎯"

---

**Usuario:** "¿Cuáles son los principales riesgos?"

**Asistente:** "⚠️ Los principales riesgos identificados en el 10-K de ALLY son:

1. **Riesgo de Crédito** (Alto) 🔴
   - Exposición a deterioro en la calidad crediticia del portafolio automotriz
   - Impacto directo en la condición financiera

2. **Riesgo de Tasas de Interés** (Medio) 🟡
   - Cambios en tasas afectan el margen de interés neto
   - El aumento de tasas podría reducir demanda de préstamos

3. **Riesgo Regulatorio** (Medio) 🟡
   - Entorno altamente regulado
   - Cambios pueden aumentar costos de cumplimiento

Estos riesgos son típicos del sector financiero pero requieren monitoreo continuo. 📋"

## Ejemplo de Reporte Generado

```markdown
# 📊 Análisis de 10-K - ALLY

## 📋 Resumen Ejecutivo
- **Riesgos Identificados**: 5 riesgos clave detectados
- **KPIs Analizados**: 10 indicadores de desempeño
- **Sentimiento General**: Positivo
- **Drivers de Ingresos**: 4 drivers principales identificados

## ⚠️ Riesgos Clave
1. Credit Risk - Alto - Crédito
2. Interest Rate Risk - Medio - Mercado
...

## 📈 Top 10 KPIs
1. Return on Equity (ROE): 14.2% - Mejorando
2. Net Interest Margin: 3.85% - Mejorando
...
```

## Limitaciones y Notas

1. **Datos de Ejemplo**: Actualmente el módulo usa datos de ejemplo para demostración. Para implementación en producción, deberías:
   - Integrar con la API oficial de SEC EDGAR
   - Implementar descarga y parsing real de documentos
   - Agregar cache de documentos

2. **Rate Limits**: Google Gemini tiene límites de tasa. Considera:
   - Implementar retry logic
   - Cachear resultados
   - Usar throttling

3. **Costos**: La API de Gemini puede tener costos asociados dependiendo del uso.

4. **Persistencia**: Las conversaciones del chatbot se mantienen en session state de Streamlit pero se reinician al refrescar la página.

## Mejoras Futuras

**Análisis SEC:**
- [ ] Integración con API real de SEC EDGAR
- [ ] Análisis de múltiples períodos para comparación
- [ ] Visualizaciones más avanzadas
- [ ] Exportación a PDF
- [ ] Análisis de tendencias históricas
- [ ] Comparación con competidores
- [ ] Alertas de cambios significativos

**Chatbot:**
- [ ] Memoria persistente entre sesiones
- [ ] Capacidad de generar gráficos bajo demanda
- [ ] Integración con búsqueda web para datos actuales
- [ ] Modo experto vs. modo simplificado
- [ ] Exportar conversaciones a PDF
- [ ] Sugerencias basadas en análisis de sentimiento
- [ ] Comparación con análisis anteriores
- [ ] Alertas proactivas sobre cambios importantes

## Soporte

Para preguntas o issues:
1. Revisa que tu API key de Gemini esté configurada correctamente
2. Verifica que todas las dependencias estén instaladas
3. Consulta los logs de error en Streamlit

## Licencia

Parte del proyecto de Valoración de Ally Financial - Grupo 8
