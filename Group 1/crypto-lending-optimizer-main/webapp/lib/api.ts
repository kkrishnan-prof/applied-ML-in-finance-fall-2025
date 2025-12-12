// API Client for Lending Rate Optimization Backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface TradeScenario {
  asset: string;
  position: 'long' | 'short';
  leverage: number;
  collateral: number;
  timestamp?: string;
}

export interface RateResponse {
  lending_rate_per_day: number;
  lending_rate_annualized: number;
  lending_rate_per_hour: number;
  base_rate: number;
  volatility_premium: number;
  predicted_volatility_24h: number;
  volatility_regime: string;
  expected_revenue: number;
  leverage_factor: number;
}

export interface SimulationResult {
  liquidation_probability: number;
  expected_pnl: number;
  total_interest_expected: number;
  pnl_percentiles: {
    '5th': number;
    '25th': number;
    '50th': number;
    '75th': number;
    '95th': number;
  };
  breakeven_price_change: number;
}

export interface RandomSample {
  timestamp: string;
  scenario: TradeScenario;
  market_conditions: {
    volatility_24h: number;
    volume_ratio: number;
    actual_future_vol: number;
  };
}

export interface BacktestRequest {
  timestamp: string;
  position: 'long' | 'short';
  leverage: number;
  collateral: number;
  duration_hours: number;
}

export interface HourlyBreakdown {
  hour: number;
  timestamp: string;
  price: number;
  equity: number;
  unrealized_pnl: number;
  interest_paid: number;
  liquidated: boolean;
}

export interface BacktestResult {
  entry_timestamp: string;
  entry_price: number;
  exit_price: number | null;
  liquidated: boolean;
  liquidation_hour: number | null;
  liquidation_price: number | null;
  liquidation_price_threshold: number;
  trader_pnl: number;
  lender_revenue: number;
  interest_charged: number;
  liquidation_fee: number;
  lending_rate_used: number;
  predicted_volatility: number;
  actual_volatility_realized: number;
  maintenance_margin: number;
  maintenance_margin_rate: number;
  hourly_breakdown: HourlyBreakdown[];
}

export interface Candle {
  index: number;
  timestamp: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
}

export interface SequentialCandlesResponse {
  candles: Candle[];
  start_timestamp: string;
  end_timestamp: string;
  dataset_index: number;
}

export const api = {
  async calculateRate(scenario: TradeScenario): Promise<RateResponse> {
    const response = await fetch(`${API_BASE_URL}/calculate_rate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scenario),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  },

  async simulateTrade(
    scenario: TradeScenario,
    duration_hours: number
  ): Promise<SimulationResult> {
    const response = await fetch(`${API_BASE_URL}/simulate_trade`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario, duration_hours }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  },

  async getRandomSamples(count: number = 5): Promise<RandomSample[]> {
    const response = await fetch(`${API_BASE_URL}/random_samples?count=${count}`);

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.samples;
  },

  async backtestTrade(request: BacktestRequest): Promise<BacktestResult> {
    const response = await fetch(`${API_BASE_URL}/backtest_trade`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  },

  async getSequentialCandles(count: number = 200): Promise<SequentialCandlesResponse> {
    const response = await fetch(`${API_BASE_URL}/sequential_candles?count=${count}`);

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  },

  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },
};
