"""
Monte Carlo Simulation Module
=============================
Simulates future stock price paths using geometric Brownian motion.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


class MonteCarloSimulation:
    """
    Monte Carlo simulation for stock price prediction.
    Uses Geometric Brownian Motion (GBM) to simulate price paths.
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        n_days: int = 252,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulation paths
            n_days: Number of trading days to simulate (252 = 1 year)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.n_days = n_days
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate_parameters(
        self, 
        prices: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate drift (mu) and volatility (sigma) from historical prices.
        
        Args:
            prices: Series of historical prices
            
        Returns:
            Tuple of (drift, volatility)
        """
        # Calculate daily returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate parameters
        mu = returns.mean()  # Daily drift
        sigma = returns.std()  # Daily volatility
        
        return mu, sigma
    
    def simulate_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float,
        dt: float = 1/252
    ) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion.
        
        GBM: dS = mu*S*dt + sigma*S*dW
        
        Args:
            S0: Initial stock price
            mu: Daily drift
            sigma: Daily volatility
            dt: Time step (1/252 for daily)
            
        Returns:
            Array of simulated price paths (n_simulations x n_days)
        """
        # Generate random walks
        Z = np.random.standard_normal((self.n_simulations, self.n_days))
        
        # GBM formula
        # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(drift + diffusion, axis=1)
        
        # Calculate price paths
        price_paths = S0 * np.exp(cumulative_returns)
        
        # Add initial price at the beginning
        price_paths = np.column_stack([np.full(self.n_simulations, S0), price_paths])
        
        return price_paths
    
    def run_simulation(
        self,
        prices: pd.Series,
        n_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on historical price data.
        
        Args:
            prices: Series of historical prices
            n_days: Override for number of days to simulate
            
        Returns:
            Dictionary with simulation results
        """
        if n_days is not None:
            self.n_days = n_days
        
        # Get current price
        S0 = prices.iloc[-1]
        
        # Calculate parameters from historical data
        mu, sigma = self.calculate_parameters(prices)
        
        # Run simulation
        price_paths = self.simulate_gbm(S0, mu, sigma)
        
        # Get final prices (end of simulation period)
        final_prices = price_paths[:, -1]
        
        # Calculate statistics
        results = self._calculate_statistics(final_prices, S0, price_paths)
        results.update({
            "initial_price": S0,
            "drift_daily": mu,
            "drift_annual": mu * 252,
            "volatility_daily": sigma,
            "volatility_annual": sigma * np.sqrt(252),
            "n_simulations": self.n_simulations,
            "n_days": self.n_days,
            "price_paths": price_paths
        })
        
        return results
    
    def _calculate_statistics(
        self,
        final_prices: np.ndarray,
        initial_price: float,
        price_paths: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate statistics from simulation results.
        
        Args:
            final_prices: Array of final prices from all simulations
            initial_price: Starting price
            price_paths: All price paths
            
        Returns:
            Dictionary with statistics
        """
        # Basic statistics
        mean_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        std_price = np.std(final_prices)
        
        # Percentiles
        percentiles = {
            5: np.percentile(final_prices, 5),
            10: np.percentile(final_prices, 10),
            25: np.percentile(final_prices, 25),
            50: np.percentile(final_prices, 50),
            75: np.percentile(final_prices, 75),
            90: np.percentile(final_prices, 90),
            95: np.percentile(final_prices, 95)
        }
        
        # Probability of profit (price above initial)
        prob_profit = np.mean(final_prices > initial_price) * 100
        
        # Value at Risk (VaR)
        var_95 = initial_price - percentiles[5]  # 95% VaR
        var_99 = initial_price - np.percentile(final_prices, 1)  # 99% VaR
        
        # Expected Shortfall (CVaR)
        cvar_95 = initial_price - np.mean(final_prices[final_prices < percentiles[5]])
        
        # Returns statistics
        returns = (final_prices - initial_price) / initial_price * 100
        expected_return = np.mean(returns)
        
        # Max and min prices across all paths
        max_price_overall = np.max(price_paths)
        min_price_overall = np.min(price_paths)
        
        return {
            "mean_final_price": mean_price,
            "median_final_price": median_price,
            "std_final_price": std_price,
            "percentiles": percentiles,
            "prob_profit": prob_profit,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "expected_return_pct": expected_return,
            "max_price": max_price_overall,
            "min_price": min_price_overall,
            "final_prices": final_prices
        }
    
    def get_confidence_intervals(
        self,
        price_paths: np.ndarray,
        confidence_levels: list = [0.90, 0.95, 0.99]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for price paths.
        
        Args:
            price_paths: Array of simulated price paths
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with lower and upper bounds for each confidence level
        """
        intervals = {}
        
        for level in confidence_levels:
            lower_pct = (1 - level) / 2 * 100
            upper_pct = (1 + level) / 2 * 100
            
            lower = np.percentile(price_paths, lower_pct, axis=0)
            upper = np.percentile(price_paths, upper_pct, axis=0)
            mean = np.mean(price_paths, axis=0)
            
            intervals[f"{int(level*100)}%"] = {
                "lower": lower,
                "upper": upper,
                "mean": mean
            }
        
        return intervals
    
    def get_summary_statistics(
        self, 
        results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame of simulation results.
        
        Args:
            results: Results from run_simulation
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = {
            "Metric": [
                "Initial Price",
                "Mean Final Price",
                "Median Final Price",
                "Std Dev of Final Price",
                "5th Percentile",
                "25th Percentile", 
                "75th Percentile",
                "95th Percentile",
                "Probability of Profit (%)",
                "Expected Return (%)",
                "VaR (95%)",
                "CVaR (95%)",
                "Daily Volatility",
                "Annual Volatility",
                "Number of Simulations",
                "Forecast Days"
            ],
            "Value": [
                f"${results['initial_price']:.2f}",
                f"${results['mean_final_price']:.2f}",
                f"${results['median_final_price']:.2f}",
                f"${results['std_final_price']:.2f}",
                f"${results['percentiles'][5]:.2f}",
                f"${results['percentiles'][25]:.2f}",
                f"${results['percentiles'][75]:.2f}",
                f"${results['percentiles'][95]:.2f}",
                f"{results['prob_profit']:.1f}%",
                f"{results['expected_return_pct']:.2f}%",
                f"${results['var_95']:.2f}",
                f"${results['cvar_95']:.2f}",
                f"{results['volatility_daily']:.4f}",
                f"{results['volatility_annual']*100:.2f}%",
                f"{results['n_simulations']:,}",
                f"{results['n_days']}"
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def calculate_option_price(
        self,
        results: Dict[str, Any],
        strike_price: float,
        option_type: str = "call",
        risk_free_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate option price using Monte Carlo simulation results.
        
        Args:
            results: Results from run_simulation
            strike_price: Strike price of the option
            option_type: "call" or "put"
            risk_free_rate: Risk-free interest rate (annual)
            
        Returns:
            Dictionary with option pricing information
        """
        final_prices = results["final_prices"]
        days = results["n_days"]
        T = days / 252  # Time to expiration in years
        
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - strike_price, 0)
        else:  # put
            payoffs = np.maximum(strike_price - final_prices, 0)
        
        # Discount to present value
        discount_factor = np.exp(-risk_free_rate * T)
        option_price = discount_factor * np.mean(payoffs)
        
        # Calculate Greeks approximation
        delta = np.mean(final_prices > strike_price) if option_type.lower() == "call" else np.mean(final_prices < strike_price)
        
        return {
            "option_type": option_type,
            "strike_price": strike_price,
            "option_price": option_price,
            "delta_approx": delta,
            "probability_itm": delta * 100
        }
