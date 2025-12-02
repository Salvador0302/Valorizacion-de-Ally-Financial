# Proyecto de Valoración de Ally Financial (ALLY)

**Análisis integral de valoración de acciones de Ally Financial usando métodos financieros clásicos y predicciones con IA.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Resumen

Este proyecto ofrece un análisis integral de valoración de la acción de Ally Financial (ALLY) utilizando varios enfoques:

### 📊 Métodos de Valoración Tradicionales
- **Valor Contable** - Valoración básica por patrimonio
- **Valor Contable Ajustado** - Patrimonio menos activos intangibles
- **Relación P/E** - Valoración basada en ganancias y comparación sectorial
- **Modelo de Descuento de Dividendos (DDM)** - Valor presente de dividendos futuros
- **Empresas Comparables** - Comparación mediante múltiplos de pares
- **DCF (Flujo de Caja Libre)** - Valoración por descuento de flujos de caja

### 🤖 Modelos de IA/ML
- **Red Neural LSTM** - Modelo de deep learning para predicción de precios
- **Simulación Monte Carlo** - Pronóstico probabilístico usando Movimiento Browniano Geométrico
- **🆕 Análisis de Reportes SEC con IA** - Análisis automático de 10-K/10-Q con Google Gemini
  - Extracción de riesgos clave
  - Identificación de KPIs principales
  - Análisis de sentimiento del MD&A
  - Identificación de drivers de ingresos
- **🆕 Chatbot Financiero Inteligente** - Asistente con IA para interpretar resultados
  - Responde preguntas sobre valoración y análisis
  - Context-aware: acceso completo a todos los datos
  - Explica conceptos financieros de manera simple
  - Proporciona recomendaciones personalizadas

## 🚀 Primeros Pasos

### Requisitos

- Python 3.9 o superior
- pip

### Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/Salvador0302/Valorizacion-de-Ally-Financial.git
cd Valorizacion-de-Ally-Financial
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. **(Nuevo)** Configura la API de Google Gemini:
   - Obtén tu API key en [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Crea un archivo `.env` en la raíz del proyecto:
   ```bash
   GEMINI_API_KEY=tu_api_key_aqui
   ```

## 💻 Uso

### Notebook de Jupyter

Ejecuta el notebook de análisis completo:
```bash
jupyter notebook notebooks/ally_valuation_analysis.ipynb
```

### Panel interactivo (Streamlit)

Inicia el panel interactivo:
```bash
streamlit run streamlit_app.py
```

El panel se abrirá en tu navegador en `http://localhost:8501`

### Uso como módulo Python

```python
from src.data_loader import DataLoader
from src.valuation import ValuationEngine
from src.lstm_model import LSTMPredictor
from src.monte_carlo import MonteCarloSimulation
from src.sec_analyzer import SECAnalyzer
from src.chatbot import ValuationChatbot

# Cargar datos
loader = DataLoader(ticker="ALLY")
summary = loader.get_summary()
prices = loader.get_historical_prices(period="5y")

# Ejecutar valoraciones
valuation = ValuationEngine(data_loader=loader)
results = valuation.get_all_valuations()
fair_value = valuation.get_fair_value_estimate()

# Simulación Monte Carlo
mc = MonteCarloSimulation(n_simulations=10000, n_days=252)
mc_results = mc.run_simulation(prices['Close'])

# Predicciones LSTM (requiere TensorFlow)
lstm = LSTMPredictor(sequence_length=60, epochs=25)
lstm.train(prices['Close'])
predictions = lstm.predict_future(prices['Close'], days_ahead=30)

# 🆕 Análisis de Reportes SEC con IA
analyzer = SECAnalyzer()
report = analyzer.generate_full_report(ticker="ALLY", filing_type="10-K")
print(report['resumen_ejecutivo'])
print(f"Riesgos identificados: {len(report['riesgos'])}")
print(f"Sentimiento: {report['sentimiento']['sentimiento_general']}")

# 🆕 Chatbot Financiero Inteligente
chatbot = ValuationChatbot()
chatbot.set_context(
    ticker="ALLY",
    current_price=summary['current_price'],
    fair_value=fair_value['fair_value_estimate'],
    valuations=results,
    mc_results=mc_results,
    sec_report=report,
    summary=summary
)

# Hacer preguntas al chatbot
respuesta = chatbot.chat("¿Es buen momento para comprar esta acción?")
print(respuesta)

# Obtener sugerencias de preguntas
sugerencias = chatbot.suggest_questions()
print("Preguntas sugeridas:", sugerencias)
```

Ver más detalles del módulo de IA en [docs/AI_MODULE.md](docs/AI_MODULE.md)

## 📁 Estructura del Proyecto

```
Valorizacion-de-Ally-Financial/
├── src/
│   ├── __init__.py           # Inicialización del paquete
│   ├── data_loader.py        # Carga de datos financieros (yfinance)
│   ├── valuation.py          # Motor de métodos de valoración
│   ├── lstm_model.py         # Modelo LSTM para predicción de precios
│   ├── monte_carlo.py        # Simulación Monte Carlo
│   ├── sec_analyzer.py       # 🆕 Análisis de reportes SEC con IA
│   └── chatbot.py            # 🆕 Chatbot financiero inteligente
├── notebooks/
│   └── ally_valuation_analysis.ipynb  # Notebook de análisis
├── docs/
│   └── AI_MODULE.md          # 🆕 Documentación del módulo de IA
├── data/                     # Carpeta de datos (cache)
├── .env                      # 🆕 Variables de entorno (API keys)
├── streamlit_app.py          # Panel Streamlit
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

## 📈 Explicación de los Métodos de Valoración

### 1. Valor Contable
- **Fórmula**: Patrimonio Total / Acciones en Circulación
- **Uso**: Estimación conservadora para empresas con muchos activos

### 2. Valor Contable Ajustado
- **Fórmula**: (Patrimonio Total - Activos Intangibles) / Acciones
- **Uso**: Estimación más conservadora excluyendo goodwill e intangibles

### 3. Relación P/E
- **Fórmula**: EPS × P/E objetivo (mediana sectorial)
- **Uso**: Valor relativo frente a pares

### 4. Modelo de Descuento de Dividendos (DDM)
- **Fórmula**: VP de dividendos de la etapa 1 + VP del valor terminal
- **Uso**: Valoración para empresas que pagan dividendos

### 5. Empresas Comparables
- **Enfoque**: Usa las medianas de P/E y P/B de empresas pares
- **Pares**: COF, SYF, DFS, AXP, C (sector servicios financieros)

### 6. DCF (Flujo de Caja Libre)
- **Fórmula**: Suma de FCF descontados + VP del valor terminal
- **Uso**: Estimación de valor intrínseco basada en fundamentales

### 7. LSTM
- **Arquitectura**: 2 capas LSTM con dropout y una capa Dense de salida
- **Uso**: Predicción basada en patrones de series temporales

### 8. Monte Carlo
- **Modelo**: Movimiento Browniano Geométrico (GBM)
- **Uso**: Distribución probabilística de precios y métricas de riesgo (VaR, CVaR)

## 📊 Funcionalidades del Panel

El panel Streamlit incluye:

- **Métricas clave**: Precio actual, capitalización, EPS, yield de dividendos
- **Gráficos interactivos**: Velas OHLC y volumen histórico
- **Pestañas de valoración**: Desglose detallado por método
- **Gráficos comparativos**: Comparación visual de resultados
- **Visualización Monte Carlo**: Rutas de precio y distribución final
- **Predicciones LSTM**: Pronóstico opcional con IA
- **Recomendación de inversión**: Señal automática de compra/mantener/venta

## 🛠️ Configuración

### Parámetros de Valoración (ajustables en el panel)

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| Rentabilidad requerida (WACC) | 10% | Tasa de descuento para DCF/DDM |
| Tasa de crecimiento (Etapa 1) | 5% | Crecimiento esperado a corto plazo |
| Crecimiento terminal | 2% | Crecimiento perpetuo a largo plazo |
| Años de pronóstico | 5 | Periodo explícito de pronóstico |

### Parámetros de Monte Carlo

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| Número de simulaciones | 10,000 | Simulaciones de trayectorias de precio |
| Días de pronóstico | 252 | Días de negociación (1 año) |

## 📚 Dependencias

- **yfinance**: API de Yahoo Finance para datos
- **pandas/numpy**: Manipulación de datos y operaciones numéricas
- **tensorflow**: Red LSTM (opcional)
- **scikit-learn**: Preprocesado
- **matplotlib/seaborn/plotly**: Visualización
- **streamlit**: Panel interactivo

## ⚠️ Aviso (Disclaimer)

Este proyecto es únicamente con fines **educativos** y no debe interpretarse como asesoría financiera. Las valoraciones de acciones implican incertidumbre y supuestos. Siempre:

- Realiza tu propia investigación
- Consulta con un asesor financiero cualificado
- Ten en cuenta que el rendimiento pasado no garantiza resultados futuros
- Considera tu tolerancia al riesgo y objetivos de inversión

## 📄 Licencia

Este proyecto es open source y está disponible bajo la licencia MIT.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Abre un Pull Request.

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en este repositorio.
