"""
FastAPI Backend for Hybrid Lending Rate Optimization System
Serves predictions to NextJS webapp for real-time trade scenario testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.rate_calculator import HybridRateCalculator
from src.data_processor import DataProcessor
from loguru import logger

# Global state
calculator: Optional[HybridRateCalculator] = None
processor: Optional[DataProcessor] = None
historical_data: Optional[pd.DataFrame] = None
feature_columns: List[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    global calculator, processor, historical_data, feature_columns

    logger.info("Starting API server...")

    try:
        # Load calculator (includes both models)
        logger.info("Loading hybrid rate calculator...")
        calculator = HybridRateCalculator()

        # Load data processor
        logger.info("Loading data processor...")
        processor = DataProcessor()

        # Load historical data
        logger.info("Loading historical data...")
        historical_data = pd.read_parquet('data/processed/processed_data.parquet')

        # Get feature columns
        feature_columns = [
            col for col in historical_data.columns
            if not any(x in col for x in ['target_', 'Open', 'High', 'Low', 'Close', 'Volume', 'log_return'])
        ]

        logger.info(f"Loaded {len(historical_data)} historical records")
        logger.info(f"Using {len(feature_columns)} features")
        logger.info("API ready!")

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

    yield

    # Cleanup (if needed)
    logger.info("Shutting down API server...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Crypto Lending Rate Optimization API",
    description="ML-powered lending rate calculator combining volatility prediction and revenue optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],  # NextJS ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TradeScenario(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "asset": "BTC",
                "position": "long",
                "leverage": 10.0,
                "collateral": 1000.0,
                "timestamp": "2024-12-01T15:00:00Z"
            }
        }
    )

    asset: str = Field(default="BTC", description="Asset symbol (BTC, ETH, etc.)")
    position: str = Field(..., description="Position type: 'long' or 'short'")
    leverage: float = Field(..., ge=1.0, le=125.0, description="Leverage ratio (1x to 125x)")
    collateral: float = Field(..., gt=0, description="Collateral amount in USD")
    timestamp: Optional[str] = Field(None, description="ISO timestamp for historical data (optional)")


class RateResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lending_rate_per_day": 0.00275,
                "lending_rate_annualized": 1.00375,
                "lending_rate_per_hour": 0.00011458,
                "base_rate": 0.0025,
                "volatility_premium": 0.00025,
                "predicted_volatility_24h": 0.65,
                "volatility_regime": "normal_vol",
                "expected_revenue": 0.000145,
                "leverage_factor": 10.0
            }
        }
    )

    lending_rate_per_day: float
    lending_rate_annualized: float
    lending_rate_per_hour: float
    base_rate: float
    volatility_premium: float
    predicted_volatility_24h: float
    volatility_regime: str
    expected_revenue: float
    leverage_factor: float


class BacktestRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-08-01 14:00:00",
                "position": "long",
                "leverage": 10.0,
                "collateral": 1000.0,
                "duration_hours": 24
            }
        }
    )

    timestamp: str = Field(..., description="Historical entry timestamp")
    position: str = Field(..., description="Position type: 'long' or 'short'")
    leverage: float = Field(..., ge=1.0, le=125.0)
    collateral: float = Field(..., gt=0)
    duration_hours: int = Field(..., ge=1, le=168)


class BacktestResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entry_timestamp": "2024-08-01 14:00:00",
                "entry_price": 50000.0,
                "exit_price": 50300.0,
                "liquidated": False,
                "liquidation_hour": None,
                "trader_pnl": -5.50,
                "lender_revenue": 30.0,
                "interest_charged": 30.0,
                "lending_rate_used": 0.00011458,
                "predicted_volatility": 0.65,
                "actual_volatility_realized": 0.62,
                "hourly_breakdown": []
            }
        }
    )

    entry_timestamp: str
    entry_price: float
    exit_price: Optional[float]
    liquidated: bool
    liquidation_hour: Optional[int]
    liquidation_price: Optional[float]
    liquidation_price_threshold: float  # Theoretical liquidation price level
    trader_pnl: float
    lender_revenue: float
    interest_charged: float
    liquidation_fee: float
    lending_rate_used: float
    predicted_volatility: float
    actual_volatility_realized: float
    maintenance_margin: float  # Dollar amount of maintenance margin
    maintenance_margin_rate: float  # MMR percentage
    hourly_breakdown: List[Dict]


class TradeSimulation(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scenario": {
                    "asset": "BTC",
                    "position": "long",
                    "leverage": 10.0,
                    "collateral": 1000.0
                },
                "duration_hours": 24
            }
        }
    )

    scenario: TradeScenario
    duration_hours: int = Field(..., ge=1, le=720, description="Trade duration in hours (max 30 days)")


class SimulationResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "liquidation_probability": 0.15,
                "expected_pnl": -25.50,
                "total_interest_expected": 27.50,
                "pnl_percentiles": {
                    "5th": -950.00,
                    "25th": -125.00,
                    "50th": -25.00,
                    "75th": 250.00,
                    "95th": 1200.00
                },
                "breakeven_price_change": 0.0028
            }
        }
    )

    liquidation_probability: float
    expected_pnl: float
    total_interest_expected: float
    pnl_percentiles: Dict[str, float]
    breakeven_price_change: float


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "1.0.0",
        "models_loaded": calculator is not None,
        "data_loaded": historical_data is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if calculator and historical_data is not None else "unhealthy",
        "calculator_loaded": calculator is not None,
        "data_loaded": historical_data is not None,
        "data_records": len(historical_data) if historical_data is not None else 0,
        "features_count": len(feature_columns)
    }


@app.post("/calculate_rate", response_model=RateResponse)
async def calculate_lending_rate(scenario: TradeScenario):
    """
    Calculate optimal lending rate for given trade scenario

    Uses current market data (latest available) unless timestamp is provided
    """
    if calculator is None or historical_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Get market data
        if scenario.timestamp:
            # Parse timestamp and find closest match
            target_time = pd.to_datetime(scenario.timestamp)
            idx = historical_data.index.searchsorted(target_time)
            if idx >= len(historical_data):
                idx = len(historical_data) - 1
        else:
            # Use most recent data
            idx = -1

        market_features = historical_data[feature_columns].iloc[idx].values

        # Calculate rate
        result = calculator.calculate_rate(
            current_market_data=market_features,
            leverage=scenario.leverage,
            asset_beta=1.0  # Default for BTC/ETH, could be parameterized
        )

        return RateResponse(
            lending_rate_per_day=result['final_rate_daily'],
            lending_rate_annualized=result['final_rate_annualized'],
            lending_rate_per_hour=result['final_rate_hourly'],
            base_rate=result['base_rate_daily'],
            volatility_premium=result['volatility_premium_daily'],
            predicted_volatility_24h=result['predicted_volatility_24h'],
            volatility_regime=result['volatility_regime'],
            expected_revenue=result['expected_revenue'],
            leverage_factor=scenario.leverage
        )

    except Exception as e:
        logger.error(f"Error calculating rate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate_trade", response_model=SimulationResult)
async def simulate_trade_outcome(simulation: TradeSimulation):
    """
    Monte Carlo simulation of trade outcome

    Simulates price paths and calculates:
    - Liquidation probability
    - Expected P&L distribution
    - Interest costs
    - Breakeven analysis
    """
    if calculator is None or historical_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        scenario = simulation.scenario

        # Get market features for rate calculation
        market_features = historical_data[feature_columns].iloc[-1].values

        # Calculate lending rate
        rate_result = calculator.calculate_rate(
            current_market_data=market_features,
            leverage=scenario.leverage
        )

        predicted_vol = rate_result['predicted_volatility_24h']
        hourly_rate = rate_result['final_rate_hourly']

        # Monte Carlo simulation
        n_simulations = 10000
        position_value = scenario.collateral * scenario.leverage

        # Calculate liquidation price
        if scenario.position == "long":
            liquidation_distance = -1.0 / scenario.leverage
        else:  # short
            liquidation_distance = 1.0 / scenario.leverage

        # Simulate price paths
        hourly_vol = predicted_vol / np.sqrt(365 * 24)
        period_vol = hourly_vol * np.sqrt(simulation.duration_hours)

        # Generate returns (geometric Brownian motion)
        simulated_returns = np.random.normal(0, period_vol, n_simulations)

        # Check liquidations
        liquidations = (
            (simulated_returns <= liquidation_distance) if scenario.position == "long"
            else (simulated_returns >= liquidation_distance)
        )

        liquidation_prob = np.mean(liquidations)

        # Calculate P&L for each path
        pnls = []
        for i, ret in enumerate(simulated_returns):
            if liquidations[i]:
                # Total loss
                pnl = -scenario.collateral
            else:
                # Price P&L
                price_pnl = position_value * ret
                # Interest cost
                interest = hourly_rate * simulation.duration_hours * position_value
                pnl = price_pnl - interest

            pnls.append(pnl)

        pnls = np.array(pnls)

        # Calculate statistics
        expected_pnl = np.mean(pnls)
        percentiles = np.percentile(pnls, [5, 25, 50, 75, 95])

        # Total interest
        total_interest = hourly_rate * simulation.duration_hours * position_value

        # Breakeven price change (including interest)
        breakeven_change = total_interest / position_value

        return SimulationResult(
            liquidation_probability=float(liquidation_prob),
            expected_pnl=float(expected_pnl),
            total_interest_expected=float(total_interest),
            pnl_percentiles={
                "5th": float(percentiles[0]),
                "25th": float(percentiles[1]),
                "50th": float(percentiles[2]),
                "75th": float(percentiles[3]),
                "95th": float(percentiles[4])
            },
            breakeven_price_change=float(breakeven_change)
        )

    except Exception as e:
        logger.error(f"Error simulating trade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market_summary")
async def get_market_summary():
    """Get current market conditions summary"""
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        latest = historical_data.iloc[-1]

        return {
            "timestamp": str(historical_data.index[-1]),
            "current_volatility_24h": float(latest.get('realized_vol_24h', 0)),
            "volume_ratio": float(latest.get('volume_ratio_24h', 1.0)),
            "trend_strength": float(latest.get('trend_strength_24h', 0)),
            "price_position": float(latest.get('price_position_24h', 0.5))
        }

    except Exception as e:
        logger.error(f"Error getting market summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rate_comparison")
async def compare_leverage_rates(asset: str = "BTC"):
    """
    Compare lending rates across different leverage levels

    Useful for showing rate schedule to users
    """
    if calculator is None or historical_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        market_features = historical_data[feature_columns].iloc[-1].values

        leverage_levels = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        rates = []

        for leverage in leverage_levels:
            result = calculator.calculate_rate(
                current_market_data=market_features,
                leverage=leverage
            )

            rates.append({
                "leverage": leverage,
                "daily_rate": result['final_rate_daily'],
                "annualized_apr": result['final_rate_annualized'],
                "base_rate": result['base_rate_daily'],
                "volatility_premium": result['volatility_premium_daily']
            })

        return {
            "asset": asset,
            "predicted_volatility": result['predicted_volatility_24h'],
            "regime": result['volatility_regime'],
            "rates_by_leverage": rates
        }

    except Exception as e:
        logger.error(f"Error comparing rates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest_trade", response_model=BacktestResult)
async def backtest_historical_trade(request: BacktestRequest):
    """
    Backtest a trade using ACTUAL historical price data

    This simulates a real trade:
    1. Enter at historical timestamp with actual price
    2. Use features UP TO entry to predict volatility & calculate rate
    3. Simulate hour-by-hour using real future candles
    4. Track liquidation path-dependently (can happen any hour)
    5. Calculate actual P&L and lender revenue

    This proves the system works with real market data!
    """
    if calculator is None or historical_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Parse entry timestamp
        entry_time = pd.to_datetime(request.timestamp)

        # Find entry index
        try:
            entry_idx = historical_data.index.get_loc(entry_time)
        except KeyError:
            # Find nearest timestamp
            entry_idx = historical_data.index.searchsorted(entry_time)
            if entry_idx >= len(historical_data):
                raise HTTPException(status_code=400, detail="Timestamp too recent, no future data")

        # Check if we have enough future data
        if entry_idx + request.duration_hours >= len(historical_data):
            raise HTTPException(
                status_code=400,
                detail=f"Not enough future data. Max duration: {len(historical_data) - entry_idx - 1} hours"
            )

        # Get entry data (what model sees)
        entry_row = historical_data.iloc[entry_idx]
        entry_features = entry_row[feature_columns].values
        entry_price = float(entry_row['Close'])
        actual_entry_time = str(historical_data.index[entry_idx])

        # Calculate lending rate using features UP TO entry
        rate_result = calculator.calculate_rate(
            current_market_data=entry_features,
            leverage=request.leverage
        )

        lending_rate_hourly = rate_result['final_rate_hourly']
        predicted_vol = rate_result['predicted_volatility_24h']

        # Initialize position
        position_size = request.collateral * request.leverage
        initial_equity = request.collateral
        current_equity = initial_equity
        total_interest = 0.0
        liquidated = False
        liquidation_hour = None
        liquidation_price = None
        hourly_breakdown = []

        # Calculate liquidation price using isolated margin formula
        # IMR (Initial Margin Rate) = 1/leverage
        # MMR (Maintenance Margin Rate) = risk-adjusted based on leverage and volatility

        imr = 1.0 / request.leverage  # Initial margin rate

        # Risk-adjusted MMR: base 2.5% + volatility premium + leverage scaling
        base_mmr = 0.025  # 2.5% base maintenance margin
        vol_adjustment = predicted_vol * 0.02  # Add 2% per unit of volatility
        leverage_scaling = min(0.01 * (request.leverage / 10), 0.05)  # Scale up to 5% for high leverage

        mmr = base_mmr + vol_adjustment + leverage_scaling  # Maintenance margin rate

        # Liquidation price formula (isolated margin):
        # Long: Liq Price = Entry × (1 - IMR + MMR)
        # Short: Liq Price = Entry × (1 + IMR - MMR)
        if request.position == "long":
            liquidation_price_theoretical = entry_price * (1 - imr + mmr)
        else:
            liquidation_price_theoretical = entry_price * (1 + imr - mmr)

        # Calculate maintenance margin in dollar terms
        # Equity at liquidation = MMR × Position Size
        maintenance_margin = mmr * position_size

        # Liquidation fee (charged when liquidated)
        liquidation_fee_rate = 0.05  # 5% of position size
        liquidation_fee = liquidation_fee_rate * position_size

        # Simulate hour-by-hour using REAL future candles
        for hour in range(1, request.duration_hours + 1):
            candle_idx = entry_idx + hour
            candle = historical_data.iloc[candle_idx]
            current_price = float(candle['Close'])

            # Calculate price P&L
            price_change_pct = (current_price - entry_price) / entry_price

            if request.position == "long":
                unrealized_pnl = position_size * price_change_pct
            else:  # short
                unrealized_pnl = -position_size * price_change_pct

            # Charge interest for this hour
            interest_this_hour = lending_rate_hourly * position_size
            total_interest += interest_this_hour

            # Current equity = initial + price P&L - interest paid
            current_equity = initial_equity + unrealized_pnl - total_interest

            # Record hourly state
            hourly_breakdown.append({
                "hour": hour,
                "timestamp": str(historical_data.index[candle_idx]),
                "price": current_price,
                "equity": float(current_equity),
                "unrealized_pnl": float(unrealized_pnl),
                "interest_paid": float(total_interest),
                "liquidated": False
            })

            # Check liquidation (equity falls below maintenance margin)
            if current_equity <= maintenance_margin:
                liquidated = True
                liquidation_hour = hour
                liquidation_price = current_price

                # Update last entry
                hourly_breakdown[-1]["liquidated"] = True

                break

        # Calculate final results
        if liquidated:
            # Trader loses all collateral
            trader_pnl = -request.collateral

            # Lender gets: interest + liquidation fee (already calculated above)
            lender_revenue = total_interest + liquidation_fee
            exit_price = liquidation_price

        else:
            # Trade completed normally
            final_candle = historical_data.iloc[entry_idx + request.duration_hours]
            exit_price = float(final_candle['Close'])

            # Trader P&L = price change - interest
            price_change_pct = (exit_price - entry_price) / entry_price
            if request.position == "long":
                price_pnl = position_size * price_change_pct
            else:
                price_pnl = -position_size * price_change_pct

            trader_pnl = price_pnl - total_interest

            # Lender gets: interest only
            lender_revenue = total_interest
            liquidation_fee = 0.0

        # Calculate actual realized volatility over the period
        future_returns = historical_data['log_return'].iloc[entry_idx+1:entry_idx+request.duration_hours+1]
        actual_vol_realized = float(future_returns.std() * np.sqrt(365 * 24))

        return BacktestResult(
            entry_timestamp=actual_entry_time,
            entry_price=entry_price,
            exit_price=exit_price,
            liquidated=liquidated,
            liquidation_hour=liquidation_hour,
            liquidation_price=liquidation_price,
            liquidation_price_threshold=float(liquidation_price_theoretical),
            trader_pnl=float(trader_pnl),
            lender_revenue=float(lender_revenue),
            interest_charged=float(total_interest),
            liquidation_fee=float(liquidation_fee if liquidated else 0.0),
            lending_rate_used=float(lending_rate_hourly),
            predicted_volatility=float(predicted_vol),
            actual_volatility_realized=actual_vol_realized,
            maintenance_margin=float(maintenance_margin),
            maintenance_margin_rate=float(mmr),
            hourly_breakdown=hourly_breakdown
        )

    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sequential_candles")
async def get_sequential_candles(count: int = 200):
    """
    Get sequential historical candles from a random start point

    This returns CONSECUTIVE hourly candles for proper backtesting visualization.
    The data will be from the test set (not training data).

    Returns:
    - timestamps: Sequential hourly timestamps
    - close_prices: Actual BTC close prices
    - start_index: Where this chunk starts in the dataset
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Use only test set data (last 20% of dataset)
        total_rows = len(historical_data)
        test_start_idx = int(total_rows * 0.8)

        # Choose random start from test set, ensuring we have enough data after
        max_start = total_rows - count - 168  # Leave room for backtesting
        start_idx = np.random.randint(test_start_idx, max_start)

        # Get sequential candles
        chunk = historical_data.iloc[start_idx:start_idx + count]

        candles = []
        for idx, (timestamp, row) in enumerate(chunk.iterrows()):
            candles.append({
                "index": idx,
                "timestamp": str(timestamp),
                "close": float(row['Close']),
                "open": float(row.get('Open', row['Close'])),
                "high": float(row.get('High', row['Close'])),
                "low": float(row.get('Low', row['Close'])),
                "volume": float(row.get('Volume', 0))
            })

        return {
            "candles": candles,
            "start_timestamp": str(chunk.index[0]),
            "end_timestamp": str(chunk.index[-1]),
            "dataset_index": start_idx
        }

    except Exception as e:
        logger.error(f"Error getting sequential candles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/random_samples")
async def get_random_samples(count: int = 5):
    """
    Get random trade scenarios from historical data for testing

    Returns 5 diverse samples (different volatility regimes)
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Get samples from different time periods and volatility regimes
        samples = []
        indices = np.random.choice(len(historical_data), size=min(count * 3, len(historical_data)), replace=False)

        for idx in indices:
            if len(samples) >= count:
                break

            row = historical_data.iloc[idx]

            # Create sample scenario
            sample = {
                "timestamp": str(historical_data.index[idx]),
                "scenario": {
                    "asset": "BTC",
                    "position": np.random.choice(["long", "short"]),
                    "leverage": float(np.random.choice([2, 5, 10, 20, 50])),
                    "collateral": float(np.random.choice([500, 1000, 2000, 5000]))
                },
                "market_conditions": {
                    "volatility_24h": float(row.get('realized_vol_24h', 0)),
                    "volume_ratio": float(row.get('volume_ratio_24h', 1.0)),
                    "actual_future_vol": float(row.get('target_vol_24h', 0))
                }
            }
            samples.append(sample)

        return {"samples": samples[:count]}

    except Exception as e:
        logger.error(f"Error getting random samples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.add("logs/api.log", rotation="10 MB")

    # Try port 8000, fallback to 8001 if busy
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except OSError:
        logger.warning("Port 8000 in use, trying 8001...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
