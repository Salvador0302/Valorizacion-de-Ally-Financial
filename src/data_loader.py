"""
Data Loader Module
==================
Loads financial data for Ally Financial using yfinance and pandas.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class DataLoader:
    """
    Loads and manages financial data for stock valuation.
    """
    
    def __init__(self, ticker: str = "ALLY"):
        """
        Initialize the DataLoader with a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (default: "ALLY")
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
        """Get stock info (cached)."""
        if self._info is None:
            self._info = self.stock.info
        return self._info
    
    def get_historical_prices(
        self, 
        period: str = "5y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            period: Time period (e.g., "1y", "5y", "max")
            interval: Data interval (e.g., "1d", "1wk", "1mo")
            
        Returns:
            DataFrame with historical prices
        """
        if self._history is None or True:  # Always refresh historical data
            self._history = self.stock.history(period=period, interval=interval)
        return self._history
    
    def get_financials(self) -> pd.DataFrame:
        """
        Get income statement data.
        
        Returns:
            DataFrame with income statement data
        """
        if self._financials is None:
            self._financials = self.stock.financials
        return self._financials
    
    def get_balance_sheet(self) -> pd.DataFrame:
        """
        Get balance sheet data.
        
        Returns:
            DataFrame with balance sheet data
        """
        if self._balance_sheet is None:
            self._balance_sheet = self.stock.balance_sheet
        return self._balance_sheet
    
    def get_cash_flow(self) -> pd.DataFrame:
        """
        Get cash flow statement data.
        
        Returns:
            DataFrame with cash flow data
        """
        if self._cash_flow is None:
            self._cash_flow = self.stock.cashflow
        return self._cash_flow
    
    def get_current_price(self) -> float:
        """
        Get the current stock price.
        
        Returns:
            Current stock price
        """
        return self.info.get("currentPrice", self.info.get("regularMarketPrice", 0))
    
    def get_shares_outstanding(self) -> int:
        """
        Get the number of shares outstanding.
        
        Returns:
            Number of shares outstanding
        """
        return self.info.get("sharesOutstanding", 0)
    
    def get_market_cap(self) -> float:
        """
        Get the market capitalization.
        
        Returns:
            Market capitalization
        """
        return self.info.get("marketCap", 0)
    
    def get_book_value_per_share(self) -> float:
        """
        Get book value per share.
        
        Returns:
            Book value per share
        """
        return self.info.get("bookValue", 0)
    
    def get_total_equity(self) -> float:
        """
        Get total stockholders' equity from balance sheet.
        
        Returns:
            Total equity
        """
        balance_sheet = self.get_balance_sheet()
        if balance_sheet.empty:
            return 0
        
        # Try different possible column names for equity
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
        Get total assets from balance sheet.
        
        Returns:
            Total assets
        """
        balance_sheet = self.get_balance_sheet()
        if balance_sheet.empty:
            return 0
            
        if "Total Assets" in balance_sheet.index:
            return float(balance_sheet.loc["Total Assets"].iloc[0])
        return 0
    
    def get_total_liabilities(self) -> float:
        """
        Get total liabilities from balance sheet.
        
        Returns:
            Total liabilities
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
        Get intangible assets from balance sheet.
        
        Returns:
            Intangible assets value
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
        Get net income from income statement.
        
        Returns:
            Net income
        """
        financials = self.get_financials()
        if financials.empty:
            return 0
            
        if "Net Income" in financials.index:
            return float(financials.loc["Net Income"].iloc[0])
        return 0
    
    def get_earnings_per_share(self) -> float:
        """
        Get trailing earnings per share.
        
        Returns:
            EPS
        """
        return self.info.get("trailingEps", 0)
    
    def get_dividend_per_share(self) -> float:
        """
        Get dividend per share.
        
        Returns:
            Dividend per share
        """
        return self.info.get("dividendRate", 0)
    
    def get_dividend_yield(self) -> float:
        """
        Get dividend yield.
        
        Returns:
            Dividend yield as decimal
        """
        return self.info.get("dividendYield", 0)
    
    def get_free_cash_flow(self) -> float:
        """
        Get free cash flow from cash flow statement.
        
        Returns:
            Free cash flow
        """
        cash_flow = self.get_cash_flow()
        if cash_flow.empty:
            return 0
            
        if "Free Cash Flow" in cash_flow.index:
            return float(cash_flow.loc["Free Cash Flow"].iloc[0])
        
        # Calculate FCF as Operating Cash Flow - Capital Expenditures
        operating_cf = 0
        capex = 0
        
        if "Operating Cash Flow" in cash_flow.index:
            operating_cf = float(cash_flow.loc["Operating Cash Flow"].iloc[0])
        
        if "Capital Expenditure" in cash_flow.index:
            capex = float(cash_flow.loc["Capital Expenditure"].iloc[0])
        
        return operating_cf + capex  # capex is typically negative
    
    def get_beta(self) -> float:
        """
        Get stock beta.
        
        Returns:
            Beta value
        """
        return self.info.get("beta", 1.0)
    
    def get_peer_tickers(self) -> list:
        """
        Get list of peer company tickers for comparable analysis.
        For Ally Financial, these are other financial services companies.
        
        Returns:
            List of peer ticker symbols
        """
        # Major financial services peers for ALLY
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
