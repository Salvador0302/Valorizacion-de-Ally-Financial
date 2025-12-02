"""
LSTM Model for Price Prediction
================================
Uses LSTM neural network to predict future stock prices.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler


class LSTMPredictor:
    """
    LSTM-based stock price predictor.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Initialize the LSTM predictor.
        
        Args:
            sequence_length: Number of time steps for input sequence
            units: Number of LSTM units
            epochs: Training epochs
            batch_size: Training batch size
        """
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.model = Sequential([
                LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(units=self.units, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            
            self.model.compile(optimizer='adam', loss='mean_squared_error')
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM predictions. Install with: pip install tensorflow")
    
    def _create_sequences(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled price data
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_data(
        self, 
        prices: pd.Series, 
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            prices: Series of historical prices
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Convert to numpy array and reshape
        data = prices.values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Split into train and test
        train_size = int(len(scaled_data) * train_ratio)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.sequence_length:]
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_data)
        X_test, y_test = self._create_sequences(test_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, y_train, X_test, y_test
    
    def train(
        self, 
        prices: pd.Series,
        train_ratio: float = 0.8,
        verbose: int = 0
    ) -> dict:
        """
        Train the LSTM model on historical prices.
        
        Args:
            prices: Series of historical prices
            train_ratio: Ratio of data to use for training
            verbose: Verbosity level for training
            
        Returns:
            Dictionary with training history
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_data(prices, train_ratio)
        
        # Build model
        self._build_model((X_train.shape[1], 1))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Evaluate model
        train_predictions = self.model.predict(X_train, verbose=0)
        test_predictions = self.model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        train_predictions = self.scaler.inverse_transform(train_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(np.mean((train_predictions - y_train_inv) ** 2))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test_inv) ** 2))
        
        return {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_predictions": train_predictions.flatten(),
            "test_predictions": test_predictions.flatten(),
            "actual_train": y_train_inv.flatten(),
            "actual_test": y_test_inv.flatten(),
            "history": history.history
        }
    
    def predict_future(
        self, 
        prices: pd.Series, 
        days_ahead: int = 30
    ) -> np.ndarray:
        """
        Predict future prices.
        
        Args:
            prices: Series of historical prices
            days_ahead: Number of days to predict
            
        Returns:
            Array of predicted prices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get the last sequence
        data = prices.values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Use the last sequence_length days
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Reshape for prediction
            input_seq = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next day
            pred = self.model.predict(input_seq, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def get_prediction_dates(
        self, 
        last_date: pd.Timestamp, 
        days_ahead: int = 30
    ) -> pd.DatetimeIndex:
        """
        Generate future dates for predictions.
        
        Args:
            last_date: Last date in historical data
            days_ahead: Number of days to predict
            
        Returns:
            DatetimeIndex of future dates
        """
        return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')
    
    def evaluate_model(
        self, 
        prices: pd.Series
    ) -> dict:
        """
        Evaluate model performance with cross-validation style metrics.
        
        Args:
            prices: Series of historical prices
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare test data
        data = prices.values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        X, y = self._create_sequences(scaled_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions)
        y_actual = self.scaler.inverse_transform(y.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - y_actual) ** 2))
        mae = np.mean(np.abs(predictions - y_actual))
        mape = np.mean(np.abs((y_actual - predictions) / y_actual)) * 100
        
        # Directional accuracy
        actual_direction = np.diff(y_actual.flatten()) > 0
        pred_direction = np.diff(predictions.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "directional_accuracy": directional_accuracy,
            "predictions": predictions.flatten(),
            "actual": y_actual.flatten()
        }
