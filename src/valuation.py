"""
Valuation Engine Module
=======================
Implements multiple stock valuation methods:
- Book Value
- Adjusted Book Value
- P/E Ratio
- Dividend Discount Model (DDM)
- Comparable Companies Analysis
- DCF (Free Cash Flow)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .data_loader import DataLoader


class ValuationEngine:
    """
    Comprehensive stock valuation engine supporting multiple methods.
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
        Calculate intrinsic value using Book Value method.
        
        Book Value = Total Equity / Shares Outstanding
        
        Returns:
            Dictionary with book value analysis
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
            "method": "Book Value",
            "intrinsic_value": final_bvps,
            "current_price": current_price,
            "total_equity": total_equity,
            "shares_outstanding": shares_outstanding,
            "price_to_book": price_to_book,
            "upside_potential": ((final_bvps - current_price) / current_price * 100) if current_price > 0 else 0
        }
    
    def adjusted_book_value_valuation(self) -> Dict[str, Any]:
        """
        Calculate intrinsic value using Adjusted Book Value method.
        
        Adjusted Book Value = (Total Equity - Intangible Assets) / Shares Outstanding
        
        This provides a more conservative estimate by excluding intangible assets.
        
        Returns:
            Dictionary with adjusted book value analysis
        """
        total_equity = self.data_loader.get_total_equity()
        intangible_assets = self.data_loader.get_intangible_assets()
        shares_outstanding = self.data_loader.get_shares_outstanding()
        current_price = self.data_loader.get_current_price()
        
        adjusted_equity = total_equity - intangible_assets
        adjusted_bvps = adjusted_equity / shares_outstanding if shares_outstanding > 0 else 0
        
        price_to_adj_book = current_price / adjusted_bvps if adjusted_bvps > 0 else 0
        
        return {
            "method": "Adjusted Book Value",
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
        Calculate intrinsic value using P/E Ratio method.
        
        Intrinsic Value = EPS * Target P/E
        
        Args:
            target_pe: Target P/E ratio (if None, uses industry average or historical)
            use_industry_average: Whether to use industry average P/E
            
        Returns:
            Dictionary with P/E analysis
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
            "method": "P/E Ratio",
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
        Calculate intrinsic value using Dividend Discount Model (DDM).
        
        For a two-stage model:
        - Stage 1: Higher growth rate for forecast_years
        - Stage 2: Terminal value with perpetual growth
        
        Args:
            required_return: Required rate of return (discount rate)
            growth_rate: Expected dividend growth rate for stage 1
            terminal_growth: Perpetual growth rate for terminal value
            forecast_years: Number of years for stage 1
            
        Returns:
            Dictionary with DDM analysis
        """
        dividend = self.data_loader.get_dividend_per_share()
        current_price = self.data_loader.get_current_price()
        
        if dividend <= 0:
            return {
                "method": "Dividend Discount Model",
                "intrinsic_value": 0,
                "current_price": current_price,
                "dividend_per_share": dividend,
                "error": "Company does not pay dividends or dividend data unavailable",
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
            "method": "Dividend Discount Model",
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
        Calculate intrinsic value using Comparable Companies Analysis.
        
        Uses peer company multiples (P/E, P/B) to estimate fair value.
        
        Returns:
            Dictionary with comparables analysis
        """
        current_price = self.data_loader.get_current_price()
        eps = self.data_loader.get_earnings_per_share()
        bvps = self.data_loader.get_book_value_per_share()
        
        peer_data = self.data_loader.get_peer_data()
        
        if peer_data.empty:
            return {
                "method": "Comparable Companies",
                "intrinsic_value": 0,
                "current_price": current_price,
                "error": "Unable to fetch peer data",
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
            "method": "Comparable Companies",
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
        Calculate intrinsic value using DCF (Free Cash Flow) method.
        
        Discounts projected free cash flows to present value.
        
        Args:
            required_return: Weighted average cost of capital (WACC)
            growth_rate: Expected FCF growth rate for stage 1
            terminal_growth: Perpetual growth rate for terminal value
            forecast_years: Number of years for explicit forecast
            
        Returns:
            Dictionary with DCF/FCF analysis
        """
        fcf = self.data_loader.get_free_cash_flow()
        shares_outstanding = self.data_loader.get_shares_outstanding()
        current_price = self.data_loader.get_current_price()
        
        if fcf <= 0:
            return {
                "method": "DCF (Free Cash Flow)",
                "intrinsic_value": 0,
                "current_price": current_price,
                "free_cash_flow": fcf,
                "error": "Negative or zero free cash flow",
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
            "method": "DCF (Free Cash Flow)",
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
