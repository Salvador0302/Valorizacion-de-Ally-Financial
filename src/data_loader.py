"""
Módulo de Carga de Datos
=========================
Carga datos financieros de Ally Financial utilizando yfinance y pandas.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class DataLoader:
    """
    Carga y gestiona datos financieros para la valoración de acciones.
    """
    
    def __init__(self, ticker: str = "ALLY"):
        """
        Inicializa el DataLoader con un símbolo ticker.
        
        Args:
            ticker: Símbolo del activo (por defecto: "ALLY")
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self._info = None
        self._history = None
        self._financials = None
        self._balance_sheet = None
        self._cash_flow = None
    
    @property
    def info(self) -> Dict[str, Any]:
        """Obtener información del activo (con caché)."""
        if self._info is None:
            self._info = self.stock.info
        return self._info
    
    def get_historical_prices(
        self, 
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Obtener datos históricos de precios.
        
        Args:
            period: Periodo de tiempo (ej.: "1y", "5y", "max")
            interval: Intervalo de datos (ej.: "1d", "1wk", "1mo")
            
        Returns:
            DataFrame con los precios históricos
        """
        if self._history is None or True:  # Always refresh historical data
            self._history = self.stock.history(period=period, interval=interval)
        return self._history
    
    def get_financials(self) -> pd.DataFrame:
        """
        Obtener datos del estado de resultados.
        
        Returns:
            DataFrame con el estado de resultados
        """
        if self._financials is None:
            self._financials = self.stock.financials
        return self._financials
    
    def get_balance_sheet(self) -> pd.DataFrame:
        """
        Obtener datos del balance general.
        
        Returns:
            DataFrame con el balance general
        """
        if self._balance_sheet is None:
            self._balance_sheet = self.stock.balance_sheet
        return self._balance_sheet
    
    def get_cash_flow(self) -> pd.DataFrame:
        """
        Obtener datos del estado de flujos de efectivo.
        
        Returns:
            DataFrame con los flujos de efectivo
        """
        if self._cash_flow is None:
            self._cash_flow = self.stock.cashflow
        return self._cash_flow
    
    def get_current_price(self) -> float:
        """
        Obtener el precio actual de la acción.
        
        Returns:
            Precio actual de la acción
        """
        return self.info.get("currentPrice", self.info.get("regularMarketPrice", 0))
    
    def get_shares_outstanding(self) -> int:
        """
        Obtener el número de acciones en circulación.
        
        Returns:
            Número de acciones en circulación
        """
        return self.info.get("sharesOutstanding", 0)
    
    def get_market_cap(self) -> float:
        """
        Obtener la capitalización de mercado.
        
        Returns:
            Capitalización de mercado
        """
        return self.info.get("marketCap", 0)
    
    def get_book_value_per_share(self) -> float:
        """
        Obtener el valor contable por acción.
        
        Returns:
            Valor contable por acción
        """
        return self.info.get("bookValue", 0)
    
    def get_total_equity(self) -> float:
        """
        Obtener el patrimonio neto total desde el balance.
        
        Returns:
            Patrimonio total
        """
        balance_sheet = self.get_balance_sheet()
        if balance_sheet.empty:
            return 0
        
        # Intentar diferentes nombres de fila que puedan contener el patrimonio
        equity_keys = [
            "Stockholders Equity",
            "Total Stockholders Equity", 
            "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ]
        
        for key in equity_keys:
            if key in balance_sheet.index:
                return float(balance_sheet.loc[key].iloc[0])
        return 0
    
    def get_total_assets(self) -> float:
        """
        Obtener los activos totales desde el balance.
        
        Returns:
            Activos totales
        """
        balance_sheet = self.get_balance_sheet()
        if balance_sheet.empty:
            return 0
            
        if "Total Assets" in balance_sheet.index:
            return float(balance_sheet.loc["Total Assets"].iloc[0])
        return 0
    
    def get_total_liabilities(self) -> float:
        """
        Obtener los pasivos totales desde el balance.
        
        Returns:
            Pasivos totales
        """
        balance_sheet = self.get_balance_sheet()
        if balance_sheet.empty:
            return 0
            
        liability_keys = [
            "Total Liabilities Net Minority Interest",
            "Total Liabilities"
        ]
        
        for key in liability_keys:
            if key in balance_sheet.index:
                return float(balance_sheet.loc[key].iloc[0])
        return 0
    
    def get_intangible_assets(self) -> float:
        """
        Obtener activos intangibles desde el balance.
        
        Returns:
            Valor de intangibles
        """
        balance_sheet = self.get_balance_sheet()
        if balance_sheet.empty:
            return 0
            
        intangible_keys = [
            "Intangible Assets",
            "Goodwill And Other Intangible Assets",
            "Goodwill"
        ]
        
        total_intangibles = 0
        for key in intangible_keys:
            if key in balance_sheet.index:
                val = balance_sheet.loc[key].iloc[0]
                if pd.notna(val):
                    total_intangibles += float(val)
        return total_intangibles
    
    def get_net_income(self) -> float:
        """
        Obtener el beneficio neto desde el estado de resultados.
        
        Returns:
            Beneficio neto
        """
        financials = self.get_financials()
        if financials.empty:
            return 0
            
        if "Net Income" in financials.index:
            return float(financials.loc["Net Income"].iloc[0])
        return 0
    
    def get_earnings_per_share(self) -> float:
        """
        Obtener las ganancias por acción (EPS) acumuladas.
        
        Returns:
            EPS
        """
        return self.info.get("trailingEps", 0)
    
    def get_dividend_per_share(self) -> float:
        """
        Obtener el dividendo por acción.
        
        Returns:
            Dividendo por acción
        """
        return self.info.get("dividendRate", 0)
    
    def get_dividend_yield(self) -> float:
        """
        Obtener el rendimiento por dividendo.
        
        Returns:
            Rendimiento por dividendo (decimal)
        """
        return self.info.get("dividendYield", 0)
    
    def get_free_cash_flow(self) -> float:
        """
        Obtener el flujo de caja libre desde el estado de flujos.
        
        Returns:
            Flujo de caja libre
        """
        cash_flow = self.get_cash_flow()
        if cash_flow.empty:
            return 0
            
        if "Free Cash Flow" in cash_flow.index:
            return float(cash_flow.loc["Free Cash Flow"].iloc[0])
        
        # Calcular FCF como Flujo operativo + CapEx (CapEx suele ser negativo)
        operating_cf = 0
        capex = 0
        
        if "Operating Cash Flow" in cash_flow.index:
            operating_cf = float(cash_flow.loc["Operating Cash Flow"].iloc[0])
        
        if "Capital Expenditure" in cash_flow.index:
            capex = float(cash_flow.loc["Capital Expenditure"].iloc[0])
        
        return operating_cf + capex  # capex is typically negative
    
    def get_beta(self) -> float:
        """
        Obtener la beta de la acción.
        
        Returns:
            Valor de beta
        """
        return self.info.get("beta", 1.0)
    
    def get_peer_tickers(self) -> list:
        """
        Obtener la lista de tickers de empresas pares para el análisis por comparables.
        Para Ally Financial, se usan otras empresas de servicios financieros.
        
        Returns:
            Lista de símbolos ticker de pares
        """
        # Principales empresas pares del sector financiero para ALLY
        return ["COF", "SYF", "DFS", "AXP", "C"]
    
    def get_peer_data(self) -> pd.DataFrame:
        """
        Get valuation metrics for peer companies.
        
        Returns:
            DataFrame with peer company metrics
        """
        peers = self.get_peer_tickers()
        peer_data = []
        
        for peer in peers:
            try:
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                peer_data.append({
                    "Ticker": peer,
                    "P/E": peer_info.get("trailingPE", None),
                    "P/B": peer_info.get("priceToBook", None),
                    "Dividend Yield": peer_info.get("dividendYield", None),
                    "Market Cap": peer_info.get("marketCap", None),
                    "ROE": peer_info.get("returnOnEquity", None)
                })
            except Exception:
                continue
        
        return pd.DataFrame(peer_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key financial metrics.
        
        Returns:
            Dictionary with key metrics
        """
        return {
            "ticker": self.ticker,
            "current_price": self.get_current_price(),
            "market_cap": self.get_market_cap(),
            "shares_outstanding": self.get_shares_outstanding(),
            "book_value_per_share": self.get_book_value_per_share(),
            "eps": self.get_earnings_per_share(),
            "dividend_per_share": self.get_dividend_per_share(),
            "dividend_yield": self.get_dividend_yield(),
            "beta": self.get_beta(),
            "total_equity": self.get_total_equity(),
            "total_assets": self.get_total_assets(),
            "net_income": self.get_net_income(),
            "free_cash_flow": self.get_free_cash_flow()
        }
