'use client';

import { useState, useEffect } from 'react';
import { api, BacktestResult } from '@/lib/api';
import Link from 'next/link';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart, Brush } from 'recharts';

interface HistoricalCandle {
  timestamp: string;
  price: number;
  index: number;
}

export default function BacktestPage() {
  // Historical price data for selection
  const [historicalData, setHistoricalData] = useState<HistoricalCandle[]>([]);
  const [loadingData, setLoadingData] = useState(false);

  // Trade parameters
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [position, setPosition] = useState<'long' | 'short'>('long');
  const [leverage, setLeverage] = useState(10);
  const [collateral, setCollateral] = useState(1000);
  const [duration, setDuration] = useState(24);

  // Results
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load random chunk of sequential historical data
  const loadRandomChunk = async () => {
    setLoadingData(true);
    setError(null);
    try {
      // Get sequential candles from test set
      const response = await api.getSequentialCandles(200);

      const candles = response.candles.map((candle) => ({
        timestamp: candle.timestamp,
        price: candle.close,
        index: candle.index
      }));

      setHistoricalData(candles);
      setSelectedIndex(100); // Start in middle
      setResult(null);
    } catch (err: any) {
      setError('Failed to load data');
    } finally {
      setLoadingData(false);
    }
  };

  useEffect(() => {
    loadRandomChunk();
  }, []);

  const handleBacktest = async () => {
    if (!historicalData[selectedIndex]) return;

    setLoading(true);
    setError(null);

    try {
      const backtestResult = await api.backtestTrade({
        timestamp: historicalData[selectedIndex].timestamp,
        position,
        leverage,
        collateral,
        duration_hours: duration,
      });

      setResult(backtestResult);
    } catch (err: any) {
      setError(err.message || 'Failed to run backtest');
    } finally {
      setLoading(false);
    }
  };

  const selectedPrice = historicalData[selectedIndex]?.price || 0;
  const selectedTimestamp = historicalData[selectedIndex]?.timestamp || '';

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            📊 Historical Backtest Simulator
          </h1>
          <p className="text-lg text-gray-300 mb-4">
            Select entry point on REAL Bitcoin price chart, then simulate the trade
          </p>
          <Link href="/" className="text-purple-400 hover:text-purple-300">
            ← Back to Rate Calculator
          </Link>
        </div>

        {!result ? (
          /* SETUP VIEW - Full width chart for entry selection */
          <div className="space-y-6">
            {/* Price Chart for Entry Selection */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-white">
                  📈 Select Entry Point on Bitcoin Price Chart
                </h2>
                <button
                  onClick={loadRandomChunk}
                  disabled={loadingData}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-2 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all shadow-lg disabled:opacity-50"
                >
                  {loadingData ? '⏳ Loading...' : '🎲 Load New Chart'}
                </button>
              </div>

              {historicalData.length > 0 && (
                <>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart
                      data={historicalData}
                      margin={{ top: 5, right: 30, left: 80, bottom: 30 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                      <XAxis
                        dataKey="index"
                        stroke="#9ca3af"
                        label={{ value: 'Hour Index', position: 'insideBottom', offset: -10, fill: '#9ca3af' }}
                        tickFormatter={(value) => `H${value}`}
                      />
                      <YAxis
                        stroke="#9ca3af"
                        domain={['auto', 'auto']}
                        tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #475569',
                          borderRadius: '8px',
                          color: '#fff'
                        }}
                        formatter={(value: any) => [`$${value.toLocaleString()}`, 'BTC Price']}
                        labelFormatter={(label) => {
                          const candle = historicalData[label as number];
                          return candle ? `Hour ${label} (${new Date(candle.timestamp).toLocaleString()})` : `Hour ${label}`;
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="price"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                      />

                      {/* Entry point marker */}
                      <ReferenceLine
                        x={selectedIndex}
                        stroke="#8b5cf6"
                        strokeWidth={3}
                        label={{ value: '▼ YOUR ENTRY', fill: '#8b5cf6', position: 'top' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  {/* Entry Selection Slider */}
                  <div className="mt-6">
                    <label className="block text-white font-medium mb-3">
                      Slide to Choose Entry Point
                    </label>
                    <input
                      type="range"
                      min="0"
                      max={historicalData.length - duration - 1}
                      value={selectedIndex}
                      onChange={(e) => setSelectedIndex(Number(e.target.value))}
                      className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                    <div className="mt-2 text-center">
                      <div className="text-3xl font-bold text-purple-300">
                        ${selectedPrice.toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-400">
                        {selectedTimestamp}
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Trade Parameters */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-xl font-bold text-white mb-4">Trade Setup</h3>

                {/* Position */}
                <div className="mb-4">
                  <label className="block text-white font-medium mb-3">Position</label>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => setPosition('long')}
                      className={`py-3 px-4 rounded-xl font-semibold transition-all ${
                        position === 'long'
                          ? 'bg-green-500 text-white shadow-lg'
                          : 'bg-white/20 text-gray-300 hover:bg-white/30'
                      }`}
                    >
                      📈 Long
                    </button>
                    <button
                      onClick={() => setPosition('short')}
                      className={`py-3 px-4 rounded-xl font-semibold transition-all ${
                        position === 'short'
                          ? 'bg-red-500 text-white shadow-lg'
                          : 'bg-white/20 text-gray-300 hover:bg-white/30'
                      }`}
                    >
                      📉 Short
                    </button>
                  </div>
                </div>

                {/* Leverage */}
                <div className="mb-4">
                  <label className="block text-white font-medium mb-3">
                    Leverage: <span className="text-purple-300">{leverage}x</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    value={leverage}
                    onChange={(e) => setLeverage(Number(e.target.value))}
                    className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                </div>

                {/* Collateral */}
                <div>
                  <label className="block text-white font-medium mb-3">
                    Collateral (USD)
                  </label>
                  <input
                    type="number"
                    value={collateral}
                    onChange={(e) => setCollateral(Number(e.target.value))}
                    className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white"
                  />
                </div>
              </div>

              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h3 className="text-xl font-bold text-white mb-4">Duration</h3>

                <div>
                  <label className="block text-white font-medium mb-3">
                    Hold Time: <span className="text-purple-300">{duration} hours</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="168"
                    value={duration}
                    onChange={(e) => setDuration(Number(e.target.value))}
                    className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                  <div className="flex justify-between text-sm text-gray-400 mt-1">
                    <span>1h</span>
                    <span>7 days</span>
                  </div>
                </div>

                <button
                  onClick={handleBacktest}
                  disabled={loading || !historicalData.length}
                  className="w-full mt-6 bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-4 px-6 rounded-xl font-bold text-lg hover:from-blue-600 hover:to-cyan-600 transition-all shadow-xl disabled:opacity-50"
                >
                  {loading ? '⏳ Running Simulation...' : '🚀 Simulate This Trade'}
                </button>
              </div>
            </div>

            {error && (
              <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200">
                ⚠️ {error}
              </div>
            )}
          </div>
        ) : (
          /* RESULTS VIEW - Full width visualization */
          <div className="space-y-6">
            {/* Back button */}
            <button
              onClick={() => setResult(null)}
              className="text-purple-400 hover:text-purple-300 font-semibold"
            >
              ← Back to Entry Selection
            </button>

            {/* Summary */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-6">
                {result.liquidated ? '💥 Position Liquidated' : '✅ Trade Completed'}
              </h2>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-500/20 p-4 rounded-xl border border-blue-400/30">
                  <div className="text-sm text-blue-200 mb-1">Entry Price</div>
                  <div className="text-2xl font-bold text-white">
                    ${result.entry_price.toLocaleString()}
                  </div>
                </div>

                <div className="bg-purple-500/20 p-4 rounded-xl border border-purple-400/30">
                  <div className="text-sm text-purple-200 mb-1">Exit Price</div>
                  <div className="text-2xl font-bold text-white">
                    ${result.exit_price?.toLocaleString() || 'N/A'}
                  </div>
                </div>

                <div className={`p-4 rounded-xl border ${
                  result.trader_pnl >= 0
                    ? 'bg-green-500/20 border-green-400/30'
                    : 'bg-red-500/20 border-red-400/30'
                }`}>
                  <div className="text-sm text-gray-200 mb-1">Trader P&L</div>
                  <div className="text-2xl font-bold text-white">
                    ${result.trader_pnl.toFixed(2)}
                  </div>
                </div>

                <div className="bg-green-500/20 p-4 rounded-xl border border-green-400/30">
                  <div className="text-sm text-green-200 mb-1">Lender Revenue</div>
                  <div className="text-2xl font-bold text-white">
                    ${result.lender_revenue.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>

            {/* Full Width Price & Equity Chart */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <h3 className="text-2xl font-bold text-white mb-6">📈 Trade Journey Visualization</h3>
              <ResponsiveContainer width="100%" height={500}>
                <ComposedChart
                  data={[
                    { hour: 0, price: result.entry_price, equity: collateral },
                    ...result.hourly_breakdown.map(h => ({
                      hour: h.hour,
                      price: h.price,
                      equity: h.equity,
                      liquidated: h.liquidated
                    }))
                  ]}
                  margin={{ top: 30, right: 100, left: 90, bottom: 30 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                  <XAxis
                    dataKey="hour"
                    stroke="#9ca3af"
                    label={{ value: 'Hours Since Entry', position: 'insideBottom', offset: -10, fill: '#9ca3af' }}
                  />
                  <YAxis
                    yAxisId="left"
                    stroke="#9ca3af"
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    stroke="#9ca3af"
                    domain={[0, 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #475569',
                      borderRadius: '8px',
                      color: '#fff'
                    }}
                    formatter={(value: any, name: string) => {
                      if (name === 'BTC Price') return [`$${value.toLocaleString()}`, 'BTC Price'];
                      if (name === 'Your Equity') return [`$${value.toFixed(2)}`, 'Equity'];
                      return [value, name];
                    }}
                  />

                  {/* Price line */}
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="price"
                    stroke="#3b82f6"
                    strokeWidth={3}
                    dot={{ r: 4, fill: '#3b82f6' }}
                    name="BTC Price"
                  />

                  {/* Equity area */}
                  <Area
                    yAxisId="right"
                    type="monotone"
                    dataKey="equity"
                    fill="#10b981"
                    fillOpacity={0.3}
                    stroke="#10b981"
                    strokeWidth={3}
                    name="Your Equity"
                  />

                  {/* Entry price */}
                  <ReferenceLine
                    yAxisId="left"
                    y={result.entry_price}
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                  />

                  {/* Exit price */}
                  {result.exit_price && (
                    <ReferenceLine
                      yAxisId="left"
                      y={result.exit_price}
                      stroke={result.liquidated ? '#ef4444' : '#10b981'}
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      label={{
                        value: result.liquidated ? 'Liquidated' : 'Exit',
                        fill: result.liquidated ? '#ef4444' : '#10b981',
                        position: 'insideBottomRight'
                      }}
                    />
                  )}

                  {/* Liquidation threshold */}
                  <ReferenceLine
                    yAxisId="left"
                    y={result.liquidation_price_threshold}
                    stroke="#fbbf24"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                  />
                </ComposedChart>
              </ResponsiveContainer>

              <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-500 rounded"></div>
                  <span className="text-gray-300">BTC Price</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-green-500 opacity-30 rounded"></div>
                  <span className="text-gray-300">Your Equity</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-0.5 bg-purple-500"></div>
                  <span className="text-gray-300">Entry: ${result.entry_price.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-0.5 bg-yellow-500"></div>
                  <span className="text-gray-300">Liq Price: ${result.liquidation_price_threshold.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                </div>
                {result.exit_price && (
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-0.5 ${result.liquidated ? 'bg-red-500' : 'bg-green-500'}`}></div>
                    <span className="text-gray-300">{result.liquidated ? 'Liquidated' : 'Exit'}: ${result.exit_price.toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h4 className="text-lg font-bold text-white mb-4">💰 Financial Outcome</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Interest Charged:</span>
                    <span className="text-white font-semibold">${result.interest_charged.toFixed(2)}</span>
                  </div>
                  {result.liquidation_fee > 0 && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Liquidation Fee:</span>
                      <span className="text-red-400 font-semibold">${result.liquidation_fee.toFixed(2)}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-gray-400">Lending Rate:</span>
                    <span className="text-white font-semibold">{(result.lending_rate_used * 100).toFixed(4)}%/hr</span>
                  </div>
                </div>
              </div>

              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h4 className="text-lg font-bold text-white mb-4">📊 Model Performance</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Predicted Vol:</span>
                    <span className="text-white font-semibold">{result.predicted_volatility.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Actual Vol:</span>
                    <span className="text-white font-semibold">{result.actual_volatility_realized.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Accuracy:</span>
                    <span className="text-purple-400 font-semibold">
                      {((1 - Math.abs(result.predicted_volatility - result.actual_volatility_realized) / result.actual_volatility_realized) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                <h4 className="text-lg font-bold text-white mb-4">⚡ Trade Details</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Position Size:</span>
                    <span className="text-white font-semibold">${(collateral * leverage).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Duration:</span>
                    <span className="text-white font-semibold">{duration} hours</span>
                  </div>
                  {result.liquidated && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Liquidated at:</span>
                      <span className="text-red-400 font-semibold">Hour {result.liquidation_hour}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
