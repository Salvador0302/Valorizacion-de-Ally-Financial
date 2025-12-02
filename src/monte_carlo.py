"""
Módulo de Simulación Monte Carlo
================================
Simula trayectorias de precios futuros usando Movimiento Browniano Geométrico.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


class MonteCarloSimulation:
    """
    Simulación Monte Carlo para la predicción de precios de acciones.
    Utiliza Movimiento Browniano Geométrico (GBM) para simular trayectorias.
    """
    
    # Standard trading days per year
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(
        self,
        n_simulations: int = 10000,
        n_days: int = 252,
        random_seed: Optional[int] = None
    ):
        """
        Inicializa el simulador Monte Carlo.
        
        Args:
            n_simulations: Número de trayectorias a simular
            n_days: Número de días de negociación a simular (252 = 1 año)
            random_seed: Semilla aleatoria para reproducibilidad
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
        Calcular deriva (mu) y volatilidad (sigma) a partir de precios históricos.
        
        Args:
            prices: Serie de precios históricos
            
        Returns:
            Tupla (drift, volatility)
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
        dt: float = None
    ) -> np.ndarray:
        """
        Simular trayectorias de precio usando Movimiento Browniano Geométrico.
        
        GBM: dS = mu*S*dt + sigma*S*dW
        
        Args:
            S0: Precio inicial
            mu: Deriva diaria
            sigma: Volatilidad diaria
            dt: Paso temporal (por defecto 1/TRADING_DAYS_PER_YEAR para datos diarios)
            
        Returns:
            Array con trayectorias simuladas (n_simulations x n_days)
        """
        if dt is None:
            dt = 1 / self.TRADING_DAYS_PER_YEAR
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
        Ejecutar simulación Monte Carlo sobre datos históricos.
        
        Args:
            prices: Serie de precios históricos
            n_days: Anular el número de días a simular
            
        Returns:
            Diccionario con los resultados de la simulación
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
            "drift_annual": mu * self.TRADING_DAYS_PER_YEAR,
            "volatility_daily": sigma,
            "volatility_annual": sigma * np.sqrt(self.TRADING_DAYS_PER_YEAR),
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
        Calcular estadísticas a partir de los resultados de la simulación.
        
        Args:
            final_prices: Array con los precios finales de todas las simulaciones
            initial_price: Precio inicial
            price_paths: Todas las trayectorias simuladas
            
        Returns:
            Diccionario con estadísticas
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
        
        # Probabilidad de ganancia (precio por encima del inicial)
        prob_profit = np.mean(final_prices > initial_price) * 100
        
        # Valor en Riesgo (VaR)
        var_95 = initial_price - percentiles[5]  # 95% VaR
        var_99 = initial_price - np.percentile(final_prices, 1)  # 99% VaR
        
        # Pérdida esperada (CVaR)
        cvar_95 = initial_price - np.mean(final_prices[final_prices < percentiles[5]])
        
        # Estadísticas de retornos
        returns = (final_prices - initial_price) / initial_price * 100
        expected_return = np.mean(returns)
        
        # Precio máximo y mínimo en todas las trayectorias
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
            "Métrica": [
                "Precio inicial",
                "Precio medio final",
                "Precio mediano final",
                "Desv. estándar (final)",
                "Percentil 5",
                "Percentil 25", 
                "Percentil 75",
                "Percentil 95",
                "Probabilidad de ganancia (%)",
                "Retorno esperado (%)",
                "VaR (95%)",
                "CVaR (95%)",
                "Volatilidad diaria",
                "Volatilidad anual",
                "Número de simulaciones",
                "Días de pronóstico"
            ],
            "Valor": [
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
        T = days / self.TRADING_DAYS_PER_YEAR  # Time to expiration in years
        
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
