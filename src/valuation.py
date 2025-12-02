"""
Módulo del Motor de Valoración
==============================
Implementa múltiples métodos de valoración de acciones:
- Valor Contable
- Valor Contable Ajustado
- Relación P/E
- Modelo de Descuento de Dividendos (DDM)
- Análisis de Empresas Comparables
- DCF (Flujo de Caja Libre)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .data_loader import DataLoader


class ValuationEngine:
    """
    Motor de valoración integral que soporta múltiples métodos.
    """
    
    def __init__(self, data_loader: Optional[DataLoader] = None, ticker: str = "ALLY"):
        """
        Initialize the valuation engine.
        
        Args:
            data_loader: Optional DataLoader instance
            ticker: Stock ticker symbol (used if data_loader not provided)
        """
        self.data_loader = data_loader or DataLoader(ticker)
        self.ticker = self.data_loader.ticker
    
    def book_value_valuation(self) -> Dict[str, Any]:
        """
        Calcular el valor intrínseco usando el método de Valor Contable.
        
        Valor Contable = Patrimonio Total / Acciones en Circulación
        
        Returns:
            Diccionario con el análisis de valor contable
        """
        total_equity = self.data_loader.get_total_equity()
        shares_outstanding = self.data_loader.get_shares_outstanding()
        current_price = self.data_loader.get_current_price()
        
        book_value_per_share = total_equity / shares_outstanding if shares_outstanding > 0 else 0
        
        # Alternative: use directly from info
        bvps_from_info = self.data_loader.get_book_value_per_share()
        
        # Use the more reliable value
        final_bvps = book_value_per_share if book_value_per_share > 0 else bvps_from_info
        
        price_to_book = current_price / final_bvps if final_bvps > 0 else 0
        
        return {
            "method": "Valor Contable",
            "intrinsic_value": final_bvps,
            "current_price": current_price,
            "total_equity": total_equity,
            "shares_outstanding": shares_outstanding,
            "price_to_book": price_to_book,
            "upside_potential": ((final_bvps - current_price) / current_price * 100) if current_price > 0 else 0
        }
    
    def adjusted_book_value_valuation(self) -> Dict[str, Any]:
        """
        Calcular el valor intrínseco usando el método de Valor Contable Ajustado.
        
        Valor Contable Ajustado = (Patrimonio Total - Activos Intangibles) / Acciones
        
        Provee una estimación más conservadora al excluir intangibles.
        
        Returns:
            Diccionario con el análisis de valor contable ajustado
        """
        total_equity = self.data_loader.get_total_equity()
        intangible_assets = self.data_loader.get_intangible_assets()
        shares_outstanding = self.data_loader.get_shares_outstanding()
        current_price = self.data_loader.get_current_price()
        
        adjusted_equity = total_equity - intangible_assets
        adjusted_bvps = adjusted_equity / shares_outstanding if shares_outstanding > 0 else 0
        
        price_to_adj_book = current_price / adjusted_bvps if adjusted_bvps > 0 else 0
        
        return {
            "method": "Valor Contable Ajustado",
            "intrinsic_value": adjusted_bvps,
            "current_price": current_price,
            "total_equity": total_equity,
            "intangible_assets": intangible_assets,
            "adjusted_equity": adjusted_equity,
            "shares_outstanding": shares_outstanding,
            "price_to_adjusted_book": price_to_adj_book,
            "upside_potential": ((adjusted_bvps - current_price) / current_price * 100) if current_price > 0 else 0
        }
    
    def pe_ratio_valuation(
        self, 
        target_pe: Optional[float] = None,
        use_industry_average: bool = True
    ) -> Dict[str, Any]:
        """
        Calcular el valor intrínseco usando la Relación P/E.
        
        Valor Intrínseco = EPS * P/E objetivo
        
        Args:
            target_pe: P/E objetivo (si None, usa mediana sectorial o histórico)
            use_industry_average: Usar promedio sectorial
            
        Returns:
            Diccionario con el análisis P/E
        """
        eps = self.data_loader.get_earnings_per_share()
        current_price = self.data_loader.get_current_price()
        current_pe = current_price / eps if eps > 0 else 0
        
        # Get industry/peer average P/E if no target specified
        if target_pe is None:
            if use_industry_average:
                peer_data = self.data_loader.get_peer_data()
                if not peer_data.empty and "P/E" in peer_data.columns:
                    peer_pes = peer_data["P/E"].dropna()
                    if len(peer_pes) > 0:
                        target_pe = peer_pes.median()
            
            if target_pe is None:
                # Use a reasonable default for financial services
                target_pe = 10.0  # Conservative estimate for banks/financial services
        
        intrinsic_value = eps * target_pe if eps > 0 else 0
        
        return {
            "method": "Relación P/E",
            "intrinsic_value": intrinsic_value,
            "current_price": current_price,
            "eps": eps,
            "current_pe": current_pe,
            "target_pe": target_pe,
            "upside_potential": ((intrinsic_value - current_price) / current_price * 100) if current_price > 0 else 0
        }
    
    def dividend_discount_model(
        self,
        required_return: float = 0.10,
        growth_rate: float = 0.03,
        terminal_growth: float = 0.02,
        forecast_years: int = 5
    ) -> Dict[str, Any]:
        """
        Calcular el valor intrínseco usando el Modelo de Descuento de Dividendos (DDM).
        
        Para un modelo de dos etapas:
        - Etapa 1: Tasa de crecimiento superior durante forecast_years
        - Etapa 2: Valor terminal con crecimiento perpetuo
        
        Args:
            required_return: Tasa requerida (descuento)
            growth_rate: Tasa de crecimiento de dividendos (etapa 1)
            terminal_growth: Tasa de crecimiento perpetua
            forecast_years: Número de años de la etapa 1
            
        Returns:
            Diccionario con el análisis DDM
        """
        dividend = self.data_loader.get_dividend_per_share()
        current_price = self.data_loader.get_current_price()
        
        if dividend <= 0:
            return {
                "method": "Modelo de Descuento de Dividendos (DDM)",
                "intrinsic_value": 0,
                "current_price": current_price,
                "dividend_per_share": dividend,
                "error": "La compañía no paga dividendos o los datos de dividendos no están disponibles",
                "upside_potential": 0
            }
        
        # Calculate present value of stage 1 dividends
        pv_stage1 = 0
        projected_dividends = []
        
        for year in range(1, forecast_years + 1):
            future_dividend = dividend * ((1 + growth_rate) ** year)
            discount_factor = (1 + required_return) ** year
            pv_dividend = future_dividend / discount_factor
            pv_stage1 += pv_dividend
            projected_dividends.append({
                "year": year,
                "dividend": future_dividend,
                "pv": pv_dividend
            })
        
        # Calculate terminal value (Gordon Growth Model)
        terminal_dividend = dividend * ((1 + growth_rate) ** forecast_years) * (1 + terminal_growth)
        terminal_value = terminal_dividend / (required_return - terminal_growth)
        pv_terminal = terminal_value / ((1 + required_return) ** forecast_years)
        
        intrinsic_value = pv_stage1 + pv_terminal
        
        return {
            "method": "Modelo de Descuento de Dividendos (DDM)",
            "intrinsic_value": intrinsic_value,
            "current_price": current_price,
            "dividend_per_share": dividend,
            "required_return": required_return,
            "growth_rate_stage1": growth_rate,
            "terminal_growth_rate": terminal_growth,
            "forecast_years": forecast_years,
            "pv_stage1_dividends": pv_stage1,
            "terminal_value": terminal_value,
            "pv_terminal_value": pv_terminal,
            "projected_dividends": projected_dividends,
            "upside_potential": ((intrinsic_value - current_price) / current_price * 100) if current_price > 0 else 0
        }
    
    def comparable_companies_valuation(self) -> Dict[str, Any]:
        """
        Calcular el valor intrínseco usando análisis de empresas comparables.
        
        Usa múltiplos de empresas pares (P/E, P/B) para estimar el valor.
        
        Returns:
            Diccionario con el análisis de comparables
        """
        current_price = self.data_loader.get_current_price()
        eps = self.data_loader.get_earnings_per_share()
        bvps = self.data_loader.get_book_value_per_share()
        
        peer_data = self.data_loader.get_peer_data()
        
        if peer_data.empty:
            return {
                "method": "Empresas Comparables",
                "intrinsic_value": 0,
                "current_price": current_price,
                "error": "No fue posible obtener datos de empresas pares",
                "upside_potential": 0
            }
        
        # Calculate median multiples
        median_pe = peer_data["P/E"].dropna().median()
        median_pb = peer_data["P/B"].dropna().median()
        
        # Implied values
        implied_value_pe = eps * median_pe if eps > 0 and pd.notna(median_pe) else 0
        implied_value_pb = bvps * median_pb if bvps > 0 and pd.notna(median_pb) else 0
        
        # Average of the two methods (weighted)
        values = []
        if implied_value_pe > 0:
            values.append(implied_value_pe)
        if implied_value_pb > 0:
            values.append(implied_value_pb)
        
        avg_implied_value = np.mean(values) if values else 0
        
        return {
            "method": "Empresas Comparables",
            "intrinsic_value": avg_implied_value,
            "current_price": current_price,
            "eps": eps,
            "book_value_per_share": bvps,
            "peer_metrics": peer_data.to_dict(orient="records"),
            "median_pe": median_pe,
            "median_pb": median_pb,
            "implied_value_pe": implied_value_pe,
            "implied_value_pb": implied_value_pb,
            "upside_potential": ((avg_implied_value - current_price) / current_price * 100) if current_price > 0 else 0
        }
    
    def dcf_fcf_valuation(
        self,
        required_return: float = 0.10,
        growth_rate: float = 0.05,
        terminal_growth: float = 0.02,
        forecast_years: int = 5
    ) -> Dict[str, Any]:
        """
        Calcular el valor intrínseco usando DCF (Flujo de Caja Libre).
        
        Descuenta los flujos de caja proyectados al valor presente.
        
        Args:
            required_return: Coste medio ponderado de capital (WACC)
            growth_rate: Tasa de crecimiento esperada del FCF (etapa 1)
            terminal_growth: Tasa perpetua de crecimiento
            forecast_years: Años de pronóstico explícito
            
        Returns:
            Diccionario con el análisis DCF/FCF
        """
        fcf = self.data_loader.get_free_cash_flow()
        shares_outstanding = self.data_loader.get_shares_outstanding()
        current_price = self.data_loader.get_current_price()
        
        if fcf <= 0:
            return {
                "method": "DCF (Flujo de Caja Libre)",
                "intrinsic_value": 0,
                "current_price": current_price,
                "free_cash_flow": fcf,
                "error": "Flujo de caja libre negativo o cero",
                "upside_potential": 0
            }
        
        # Calculate present value of projected FCFs
        pv_fcfs = 0
        projected_fcfs = []
        
        for year in range(1, forecast_years + 1):
            future_fcf = fcf * ((1 + growth_rate) ** year)
            discount_factor = (1 + required_return) ** year
            pv_fcf = future_fcf / discount_factor
            pv_fcfs += pv_fcf
            projected_fcfs.append({
                "year": year,
                "fcf": future_fcf,
                "pv": pv_fcf
            })
        
        # Terminal value (Gordon Growth Model for FCF)
        terminal_fcf = fcf * ((1 + growth_rate) ** forecast_years) * (1 + terminal_growth)
        terminal_value = terminal_fcf / (required_return - terminal_growth)
        pv_terminal = terminal_value / ((1 + required_return) ** forecast_years)
        
        # Enterprise value
        enterprise_value = pv_fcfs + pv_terminal
        
        # Equity value per share
        equity_value_per_share = enterprise_value / shares_outstanding if shares_outstanding > 0 else 0
        
        return {
            "method": "DCF (Flujo de Caja Libre)",
            "intrinsic_value": equity_value_per_share,
            "current_price": current_price,
            "free_cash_flow": fcf,
            "wacc": required_return,
            "growth_rate": growth_rate,
            "terminal_growth_rate": terminal_growth,
            "forecast_years": forecast_years,
            "pv_projected_fcfs": pv_fcfs,
            "terminal_value": terminal_value,
            "pv_terminal_value": pv_terminal,
            "enterprise_value": enterprise_value,
            "shares_outstanding": shares_outstanding,
            "projected_fcfs": projected_fcfs,
            "upside_potential": ((equity_value_per_share - current_price) / current_price * 100) if current_price > 0 else 0
        }
    
    def get_all_valuations(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all valuation methods and return a comprehensive comparison.
        
        Returns:
            Dictionary with all valuation results
        """
        return {
            "book_value": self.book_value_valuation(),
            "adjusted_book_value": self.adjusted_book_value_valuation(),
            "pe_ratio": self.pe_ratio_valuation(),
            "dividend_discount": self.dividend_discount_model(),
            "comparables": self.comparable_companies_valuation(),
            "dcf_fcf": self.dcf_fcf_valuation()
        }
    
    def get_valuation_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame comparing all valuation methods.
        
        Returns:
            DataFrame with valuation comparison
        """
        valuations = self.get_all_valuations()
        
        summary_data = []
        for method_name, result in valuations.items():
            summary_data.append({
                "Method": result.get("method", method_name),
                "Intrinsic Value ($)": round(result.get("intrinsic_value", 0), 2),
                "Current Price ($)": round(result.get("current_price", 0), 2),
                "Upside Potential (%)": round(result.get("upside_potential", 0), 2),
                "Error": result.get("error", "")
            })
        
        return pd.DataFrame(summary_data)
    
    def get_fair_value_estimate(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate a weighted average fair value estimate.
        
        Args:
            weights: Optional dictionary of weights for each method
            
        Returns:
            Dictionary with weighted fair value estimate
        """
        if weights is None:
            # Default equal weights for valid methods
            weights = {
                "book_value": 0.15,
                "adjusted_book_value": 0.10,
                "pe_ratio": 0.20,
                "dividend_discount": 0.20,
                "comparables": 0.15,
                "dcf_fcf": 0.20
            }
        
        valuations = self.get_all_valuations()
        
        weighted_sum = 0
        total_weight = 0
        valid_methods = []
        
        for method, weight in weights.items():
            if method in valuations:
                value = valuations[method].get("intrinsic_value", 0)
                error = valuations[method].get("error", None)
                
                if value > 0 and error is None:
                    weighted_sum += value * weight
                    total_weight += weight
                    valid_methods.append(method)
        
        fair_value = weighted_sum / total_weight if total_weight > 0 else 0
        current_price = self.data_loader.get_current_price()
        
        return {
            "fair_value_estimate": fair_value,
            "current_price": current_price,
            "upside_potential": ((fair_value - current_price) / current_price * 100) if current_price > 0 else 0,
            "valid_methods_used": valid_methods,
            "total_weight": total_weight
        }
