"""
Real Data Loader

Loads and preprocesses real market data for experiments.
Supports multiple data sources and frequencies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf


class RealDataLoader:
    """
    Loader for real market data from various sources.

    Supports:
    - Yahoo Finance (stocks, crypto, indices)
    - CSV files
    - Multiple frequencies (tick, minute, hourly, daily)
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        verbose: bool = True
    ):
        """
        Initialize the real data loader.

        Args:
            cache_dir: Directory to cache downloaded data
            verbose: Whether to print detailed logs
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def load_from_yahoo(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD', '^GSPC')
            start_date: Start date
            end_date: End date
            interval: Data interval ('1m', '5m', '1h', '1d', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if self.verbose:
            print(f"Loading {symbol} from Yahoo Finance ({start_date} to {end_date}, {interval})")

        try:
            # Download data
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True  # Explicitly set to avoid FutureWarning
            )

            # Validate downloaded data
            if df.empty:
                raise ValueError(f"No data returned for {symbol} (empty DataFrame)")

            # Check if dataframe has columns (yfinance sometimes returns empty columns)
            if len(df.columns) == 0:
                raise ValueError(f"No columns in downloaded data for {symbol}")

            # Handle MultiIndex columns (yfinance sometimes returns this)
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                df.columns = df.columns.get_level_values(0)

            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")

            # Remove any rows with all NaN values
            df = df.dropna(how='all')

            if len(df) == 0:
                raise ValueError(f"No valid data for {symbol} after removing NaN rows")

            # Cache the data
            cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.csv"
            df.to_csv(cache_file)

            if self.verbose:
                print(f"  Loaded {len(df)} data points")
                print(f"  Cached to: {cache_file}")

            return df

        except Exception as e:
            print(f"Error loading data from Yahoo Finance for {symbol}: {e}")
            raise

    def load_from_csv(
        self,
        filepath: str,
        date_column: str = 'timestamp',
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            date_column: Name of the date/timestamp column
            parse_dates: Whether to parse dates

        Returns:
            DataFrame with market data
        """
        if self.verbose:
            print(f"Loading data from CSV: {filepath}")

        try:
            df = pd.read_csv(filepath)

            if parse_dates and date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.set_index(date_column)

            if self.verbose:
                print(f"  Loaded {len(df)} data points")
                print(f"  Columns: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            raise

    def load_multiple_assets(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple assets.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}

        for symbol in symbols:
            try:
                df = self.load_from_yahoo(symbol, start_date, end_date, interval)
                data[symbol] = df
            except Exception as e:
                print(f"Warning: Failed to load {symbol}: {e}")

        if len(data) == 0:
            raise ValueError(f"Failed to load any assets from {symbols}")

        if self.verbose:
            print(f"\nSuccessfully loaded {len(data)}/{len(symbols)} assets")

        return data

    def prepare_hft_data(
        self,
        df: pd.DataFrame,
        lookback_window: int = 100,
        add_noise: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for HFT agent (tick-level with order book features).

        Args:
            df: DataFrame with OHLCV data
            lookback_window: Number of historical ticks to include
            add_noise: Whether to add synthetic order book noise

        Returns:
            Dictionary with prepared data
        """
        if self.verbose:
            print("Preparing HFT data...")

        # Use Close prices as mid-price
        mid_prices = df['Close'].values

        # Generate synthetic order book data
        # In real implementation, this would come from actual order book data
        bid_prices = mid_prices * (1 - np.random.uniform(0.0001, 0.001, size=(len(mid_prices), 10)))
        ask_prices = mid_prices[:, np.newaxis] * (1 + np.random.uniform(0.0001, 0.001, size=(len(mid_prices), 10)))

        # Generate synthetic volumes
        volumes = df['Volume'].values[:, np.newaxis] * np.random.uniform(0.1, 1.0, size=(len(mid_prices), 10))
        bid_volumes = volumes
        ask_volumes = volumes * np.random.uniform(0.8, 1.2, size=volumes.shape)

        # Compute technical indicators
        returns = np.diff(mid_prices) / mid_prices[:-1]
        volatility = pd.Series(returns).rolling(window=20).std().values

        # Pad to match length
        returns = np.concatenate([[0], returns])
        volatility = np.concatenate([volatility[:1], volatility])
        volatility = np.nan_to_num(volatility, nan=0.0)

        return {
            'mid_prices': mid_prices,
            'bid_prices': bid_prices,
            'ask_prices': ask_prices,
            'bid_volumes': bid_volumes,
            'ask_volumes': ask_volumes,
            'returns': returns,
            'volatility': volatility,
            'timestamps': df.index.values if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df))
        }

    def prepare_mft_data(
        self,
        df: pd.DataFrame,
        lookback_window: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for MFT agent (hourly OHLCV with technical indicators).

        Args:
            df: DataFrame with OHLCV data
            lookback_window: Number of historical periods

        Returns:
            Dictionary with prepared data
        """
        if self.verbose:
            print("Preparing MFT data...")

        # Extract OHLCV
        ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

        # Compute technical indicators
        close = df['Close'].values

        # Simple Moving Average
        sma_20 = pd.Series(close).rolling(window=20).mean().values
        sma_50 = pd.Series(close).rolling(window=50).mean().values

        # RSI
        rsi = self._compute_rsi(close, period=14)

        # MACD
        macd, signal = self._compute_macd(close)

        # Fill NaN values
        sma_20 = np.nan_to_num(sma_20, nan=close[0])
        sma_50 = np.nan_to_num(sma_50, nan=close[0])
        rsi = np.nan_to_num(rsi, nan=50.0)
        macd = np.nan_to_num(macd, nan=0.0)
        signal = np.nan_to_num(signal, nan=0.0)

        return {
            'ohlcv': ohlcv,
            'close': close,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': signal,
            'timestamps': df.index.values if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df))
        }

    def prepare_lft_data(
        self,
        data: Dict[str, pd.DataFrame],
        lookback_window: int = 252
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for LFT agent (daily multi-asset portfolio data).

        Args:
            data: Dictionary mapping symbols to DataFrames
            lookback_window: Number of historical days

        Returns:
            Dictionary with prepared data
        """
        if self.verbose:
            print(f"Preparing LFT data for {len(data)} assets...")

        # Align all data by date
        symbols = list(data.keys())
        dfs = [data[s][['Close']].rename(columns={'Close': s}) for s in symbols]

        # Merge on index
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.join(df, how='inner')

        # Compute returns matrix
        returns = combined.pct_change().values
        returns = np.nan_to_num(returns, nan=0.0)

        # Compute correlation matrix (rolling)
        correlations = []
        for i in range(lookback_window, len(returns)):
            window_returns = returns[i - lookback_window:i]
            corr = np.corrcoef(window_returns.T)
            corr = np.nan_to_num(corr, nan=0.0)
            correlations.append(corr)

        correlations = np.array(correlations)

        return {
            'prices': combined.values,
            'returns': returns,
            'correlations': correlations,
            'symbols': symbols,
            'timestamps': combined.index.values
        }

    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = pd.Series(gains).rolling(window=period).mean().values
        avg_losses = pd.Series(losses).rolling(window=period).mean().values

        # Avoid division by zero
        avg_losses = np.where(avg_losses == 0, 1e-8, avg_losses)

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Pad to match original length
        rsi = np.concatenate([[50.0], rsi])

        return rsi

    def _compute_macd(
        self,
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute MACD indicator."""
        # Exponential moving averages
        ema_fast = pd.Series(prices).ewm(span=fast_period).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow_period).mean().values

        # MACD line
        macd = ema_fast - ema_slow

        # Signal line
        signal = pd.Series(macd).ewm(span=signal_period).mean().values

        return macd, signal

    def get_crisis_data(
        self,
        crisis_name: str = "covid_2020"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load predefined crisis period data.

        Args:
            crisis_name: Name of crisis period
                - 'covid_2020': COVID-19 crash (Feb-Mar 2020)
                - 'financial_2008': Financial crisis (Sep-Nov 2008)
                - 'dotcom_2000': Dot-com bubble burst (Mar-Oct 2000)

        Returns:
            Dictionary with market data for crisis period
        """
        crisis_periods = {
            'covid_2020': {
                'start': '2020-02-01',
                'end': '2020-04-30',
                'symbols': ['^GSPC', 'AAPL', 'MSFT', 'BTC-USD']
            },
            'financial_2008': {
                'start': '2008-09-01',
                'end': '2008-12-31',
                'symbols': ['^GSPC', 'GS', 'BAC', 'GLD']
            },
            'dotcom_2000': {
                'start': '2000-03-01',
                'end': '2000-12-31',
                'symbols': ['^GSPC', '^IXIC', 'MSFT', 'CSCO']
            }
        }

        if crisis_name not in crisis_periods:
            raise ValueError(f"Unknown crisis: {crisis_name}. Available: {list(crisis_periods.keys())}")

        period = crisis_periods[crisis_name]

        if self.verbose:
            print(f"\nLoading {crisis_name.upper()} data:")
            print(f"  Period: {period['start']} to {period['end']}")
            print(f"  Assets: {period['symbols']}")

        return self.load_multiple_assets(
            symbols=period['symbols'],
            start_date=period['start'],
            end_date=period['end'],
            interval='1d'
        )

    def get_bull_market_data(
        self,
        period_name: str = "post_covid_2020"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load predefined bull market period data.

        Args:
            period_name: Name of bull market period
                - 'post_covid_2020': Post-COVID rally (May 2020 - Dec 2021)
                - 'recovery_2009': Post-crisis recovery (Mar 2009 - Dec 2010)

        Returns:
            Dictionary with market data for bull market period
        """
        bull_periods = {
            'post_covid_2020': {
                'start': '2020-05-01',
                'end': '2021-12-31',
                'symbols': ['^GSPC', 'AAPL', 'TSLA', 'BTC-USD']
            },
            'recovery_2009': {
                'start': '2009-03-01',
                'end': '2010-12-31',
                'symbols': ['^GSPC', 'AAPL', 'AMZN', 'GLD']
            }
        }

        if period_name not in bull_periods:
            raise ValueError(f"Unknown bull period: {period_name}. Available: {list(bull_periods.keys())}")

        period = bull_periods[period_name]

        if self.verbose:
            print(f"\nLoading {period_name.upper()} data:")
            print(f"  Period: {period['start']} to {period['end']}")
            print(f"  Assets: {period['symbols']}")

        return self.load_multiple_assets(
            symbols=period['symbols'],
            start_date=period['start'],
            end_date=period['end'],
            interval='1d'
        )

    def get_stable_market_data(
        self,
        year: int = 2019
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stable market period data.

        Args:
            year: Year for stable period (default: 2019 - pre-COVID)

        Returns:
            Dictionary with market data for stable period
        """
        symbols = ['^GSPC', 'AAPL', 'MSFT', 'JNJ', 'PG']

        if self.verbose:
            print(f"\nLoading STABLE MARKET data for {year}:")
            print(f"  Period: {year}-01-01 to {year}-12-31")
            print(f"  Assets: {symbols}")

        return self.load_multiple_assets(
            symbols=symbols,
            start_date=f'{year}-01-01',
            end_date=f'{year}-12-31',
            interval='1d'
        )
