"""
Módulo de Análisis de Ventas (Revenue) para Ally Financial.
Incluye análisis histórico, proyecciones y forecasting con modelos de ML.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from datetime import datetime, timedelta


class RevenueAnalyzer:
    """Analizador de ventas y revenue para empresas financieras."""
    
    def __init__(self, ticker: str = "ALLY"):
        """
        Inicializar el analizador de revenue.
        
        Args:
            ticker: Símbolo de la empresa
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.revenue_data = None
        self.quarterly_revenue = None
        self.annual_revenue = None
        
    def fetch_revenue_data(self) -> Dict:
        """
        Obtener datos históricos de revenue de la empresa.
        
        Returns:
            Diccionario con datos de revenue
        """
        try:
            # Obtener datos financieros
            financials = self.stock.financials
            quarterly_financials = self.stock.quarterly_financials
            
            # Extraer revenue (Total Revenue o Operating Revenue)
            if 'Total Revenue' in financials.index:
                self.annual_revenue = financials.loc['Total Revenue'].sort_index()
            elif 'Operating Revenue' in financials.index:
                self.annual_revenue = financials.loc['Operating Revenue'].sort_index()
            
            if 'Total Revenue' in quarterly_financials.index:
                self.quarterly_revenue = quarterly_financials.loc['Total Revenue'].sort_index()
            elif 'Operating Revenue' in quarterly_financials.index:
                self.quarterly_revenue = quarterly_financials.loc['Operating Revenue'].sort_index()
            
            return {
                'annual': self.annual_revenue,
                'quarterly': self.quarterly_revenue,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def calculate_growth_rates(self) -> Dict:
        """
        Calcular tasas de crecimiento de revenue.
        
        Returns:
            Diccionario con métricas de crecimiento
        """
        if self.annual_revenue is None or len(self.annual_revenue) < 2:
            return {'error': 'Datos insuficientes para calcular crecimiento'}
        
        # Crecimiento año a año (YoY)
        yoy_growth = self.annual_revenue.pct_change() * 100
        
        # Crecimiento trimestral
        qoq_growth = None
        if self.quarterly_revenue is not None and len(self.quarterly_revenue) >= 2:
            qoq_growth = self.quarterly_revenue.pct_change() * 100
        
        # CAGR (Compound Annual Growth Rate)
        years = len(self.annual_revenue) - 1
        if years > 0:
            initial = self.annual_revenue.iloc[0]
            final = self.annual_revenue.iloc[-1]
            cagr = ((final / initial) ** (1 / years) - 1) * 100
        else:
            cagr = 0
        
        # Crecimiento promedio
        avg_growth = yoy_growth.mean()
        
        # Volatilidad del crecimiento
        growth_volatility = yoy_growth.std()
        
        return {
            'cagr': cagr,
            'avg_annual_growth': avg_growth,
            'growth_volatility': growth_volatility,
            'yoy_growth': yoy_growth.to_dict(),
            'qoq_growth': qoq_growth.to_dict() if qoq_growth is not None else None,
            'latest_growth': yoy_growth.iloc[-1] if len(yoy_growth) > 0 else 0,
            'success': True
        }
    
    def analyze_revenue_components(self) -> Dict:
        """
        Analizar componentes y segmentos de revenue.
        
        Returns:
            Diccionario con análisis de componentes
        """
        try:
            # Obtener income statement completo
            income_stmt = self.stock.financials
            
            components = {}
            
            # Buscar diferentes líneas de revenue
            revenue_lines = [
                'Total Revenue',
                'Operating Revenue',
                'Net Interest Income',
                'Non Interest Income',
                'Interest Income',
                'Other Revenue'
            ]
            
            for line in revenue_lines:
                if line in income_stmt.index:
                    components[line] = income_stmt.loc[line].sort_index()
            
            # Calcular composición del revenue
            total_rev = components.get('Total Revenue') or components.get('Operating Revenue')
            
            composition = {}
            if total_rev is not None:
                for key, value in components.items():
                    if key not in ['Total Revenue', 'Operating Revenue']:
                        try:
                            # Calcular porcentaje del total
                            pct = (value / total_rev * 100).iloc[-1]
                            composition[key] = {
                                'amount': value.iloc[-1],
                                'percentage': pct
                            }
                        except:
                            pass
            
            return {
                'components': {k: v.to_dict() for k, v in components.items()},
                'composition': composition,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def forecast_revenue(self, periods: int = 4, method: str = 'linear') -> Dict:
        """
        Proyectar revenue futuro usando diferentes métodos.
        
        Args:
            periods: Número de períodos a proyectar
            method: Método de proyección ('linear', 'average_growth', 'cagr')
            
        Returns:
            Diccionario con proyecciones
        """
        if self.quarterly_revenue is None or len(self.quarterly_revenue) < 4:
            return {'error': 'Datos insuficientes para proyección'}
        
        historical_data = self.quarterly_revenue.values
        
        projections = []
        
        if method == 'linear':
            # Regresión lineal simple
            x = np.arange(len(historical_data))
            coeffs = np.polyfit(x, historical_data, 1)
            
            for i in range(periods):
                next_x = len(historical_data) + i
                projected = coeffs[0] * next_x + coeffs[1]
                projections.append(projected)
        
        elif method == 'average_growth':
            # Usar crecimiento promedio
            growth_rates = pd.Series(historical_data).pct_change().dropna()
            avg_growth = growth_rates.mean()
            
            last_value = historical_data[-1]
            for i in range(periods):
                projected = last_value * (1 + avg_growth) ** (i + 1)
                projections.append(projected)
        
        elif method == 'cagr':
            # Usar CAGR
            growth_metrics = self.calculate_growth_rates()
            cagr = growth_metrics.get('cagr', 0) / 100
            
            last_value = historical_data[-1]
            for i in range(periods):
                projected = last_value * (1 + cagr / 4) ** (i + 1)  # Dividir por 4 para quarterly
                projections.append(projected)
        
        # Calcular intervalos de confianza (simple)
        std_dev = pd.Series(historical_data).std()
        
        return {
            'projections': projections,
            'method': method,
            'periods': periods,
            'confidence_intervals': {
                'lower': [p - 1.96 * std_dev for p in projections],
                'upper': [p + 1.96 * std_dev for p in projections]
            },
            'success': True
        }
    
    def identify_revenue_drivers(self) -> Dict:
        """
        Identificar los principales drivers de revenue.
        
        Returns:
            Diccionario con drivers identificados
        """
        drivers = []
        
        # Para empresas financieras como Ally, los drivers típicos son:
        drivers_info = [
            {
                'driver': 'Net Interest Income',
                'description': 'Diferencia entre intereses ganados y pagados',
                'importance': 'Alta',
                'trend': 'Variable según tasas de interés'
            },
            {
                'driver': 'Loan Originations',
                'description': 'Volumen de nuevos préstamos otorgados',
                'importance': 'Alta',
                'trend': 'Dependiente de demanda automotriz'
            },
            {
                'driver': 'Interest Rates',
                'description': 'Tasas de interés de mercado',
                'importance': 'Alta',
                'trend': 'Afecta margen de interés neto'
            },
            {
                'driver': 'Fee Income',
                'description': 'Ingresos por servicios y comisiones',
                'importance': 'Media',
                'trend': 'Complementario al ingreso por intereses'
            },
            {
                'driver': 'Insurance Premiums',
                'description': 'Primas de seguros automotrices',
                'importance': 'Media',
                'trend': 'Crecimiento estable'
            }
        ]
        
        # Intentar obtener métricas reales
        try:
            income_stmt = self.stock.financials
            
            for driver_info in drivers_info:
                driver_name = driver_info['driver']
                
                if driver_name in income_stmt.index:
                    values = income_stmt.loc[driver_name].sort_index()
                    
                    # Calcular tendencia
                    if len(values) >= 2:
                        recent_growth = ((values.iloc[-1] - values.iloc[-2]) / values.iloc[-2] * 100)
                        driver_info['recent_value'] = values.iloc[-1]
                        driver_info['recent_growth'] = recent_growth
                        
                        if recent_growth > 5:
                            driver_info['trend'] = 'Creciendo'
                        elif recent_growth < -5:
                            driver_info['trend'] = 'Declinando'
                        else:
                            driver_info['trend'] = 'Estable'
                
                drivers.append(driver_info)
        
        except:
            drivers = drivers_info
        
        return {
            'drivers': drivers,
            'success': True
        }
    
    def analyze_seasonality(self) -> Dict:
        """
        Analizar patrones de estacionalidad en el revenue.
        
        Returns:
            Diccionario con análisis de estacionalidad
        """
        if self.quarterly_revenue is None or len(self.quarterly_revenue) < 8:
            return {'error': 'Datos insuficientes para análisis de estacionalidad'}
        
        # Convertir a DataFrame
        df = pd.DataFrame({
            'revenue': self.quarterly_revenue.values,
            'quarter': [(i % 4) + 1 for i in range(len(self.quarterly_revenue))]
        })
        
        # Promediar por quarter
        seasonal_pattern = df.groupby('quarter')['revenue'].mean()
        
        # Calcular índice de estacionalidad
        overall_mean = seasonal_pattern.mean()
        seasonal_index = (seasonal_pattern / overall_mean * 100).to_dict()
        
        # Identificar quarter más fuerte y más débil
        strongest_quarter = seasonal_pattern.idxmax()
        weakest_quarter = seasonal_pattern.idxmin()
        
        return {
            'seasonal_pattern': seasonal_pattern.to_dict(),
            'seasonal_index': seasonal_index,
            'strongest_quarter': int(strongest_quarter),
            'weakest_quarter': int(weakest_quarter),
            'seasonality_strength': seasonal_pattern.std() / overall_mean * 100,
            'success': True
        }
    
    def get_revenue_summary(self) -> Dict:
        """
        Obtener resumen completo del análisis de revenue.
        
        Returns:
            Diccionario con resumen ejecutivo
        """
        # Fetch data si no está disponible
        if self.annual_revenue is None:
            self.fetch_revenue_data()
        
        # Calcular todas las métricas
        growth_metrics = self.calculate_growth_rates()
        components = self.analyze_revenue_components()
        drivers = self.identify_revenue_drivers()
        seasonality = self.analyze_seasonality()
        forecast = self.forecast_revenue(periods=4, method='cagr')
        
        # Revenue actual
        latest_annual = self.annual_revenue.iloc[-1] if self.annual_revenue is not None else 0
        latest_quarterly = self.quarterly_revenue.iloc[-1] if self.quarterly_revenue is not None else 0
        
        # TTM (Trailing Twelve Months)
        ttm_revenue = self.quarterly_revenue.iloc[-4:].sum() if self.quarterly_revenue is not None and len(self.quarterly_revenue) >= 4 else latest_annual
        
        summary = {
            'ticker': self.ticker,
            'latest_annual_revenue': latest_annual,
            'latest_quarterly_revenue': latest_quarterly,
            'ttm_revenue': ttm_revenue,
            'growth_metrics': growth_metrics,
            'revenue_components': components,
            'revenue_drivers': drivers,
            'seasonality_analysis': seasonality,
            'revenue_forecast': forecast,
            'annual_data': self.annual_revenue.to_dict() if self.annual_revenue is not None else {},
            'quarterly_data': self.quarterly_revenue.to_dict() if self.quarterly_revenue is not None else {}
        }
        
        return summary


def format_currency(value: float) -> str:
    """Formatear valor como moneda."""
    if abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"
