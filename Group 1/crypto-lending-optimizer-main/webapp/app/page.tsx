'use client';

import { useState, useEffect } from 'react';
import { api, TradeScenario, RateResponse, SimulationResult, RandomSample } from '@/lib/api';
import Link from 'next/link';

export default function Home() {
  const [scenario, setScenario] = useState<TradeScenario>({
    asset: 'BTC',
    position: 'long',
    leverage: 10,
    collateral: 1000,
  });

  const [duration, setDuration] = useState(24);
  const [rateData, setRateData] = useState<RateResponse | null>(null);
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [randomSamples, setRandomSamples] = useState<RandomSample[]>([]);

  // Load random samples on mount
  useEffect(() => {
    loadRandomSamples();
  }, []);

  const loadRandomSamples = async () => {
    try {
      const samples = await api.getRandomSamples(5);
      setRandomSamples(samples);
    } catch (err) {
      console.error('Failed to load samples:', err);
    }
  };

  const handleCalculate = async () => {
    setLoading(true);
    setError(null);

    try {
      const [rate, sim] = await Promise.all([
        api.calculateRate(scenario),
        api.simulateTrade(scenario, duration),
      ]);

      setRateData(rate);
      setSimulation(sim);
    } catch (err: any) {
      setError(err.message || 'Failed to calculate');
    } finally {
      setLoading(false);
    }
  };

  const loadRandomScenario = () => {
    if (randomSamples.length === 0) return;

    const sample = randomSamples[Math.floor(Math.random() * randomSamples.length)];
    setScenario({
      ...sample.scenario,
      timestamp: sample.timestamp,
    });
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            🚀 Crypto Lending Rate Optimizer
          </h1>
          <p className="text-xl text-gray-300 mb-4">
            ML-Powered Dynamic Rate Calculator for Bitcoin Margin Trading
          </p>
          <Link
            href="/backtest"
            className="inline-block bg-gradient-to-r from-cyan-500 to-blue-500 text-white px-6 py-3 rounded-xl font-semibold hover:from-cyan-600 hover:to-blue-600 transition-all shadow-lg"
          >
            📊 Try Historical Backtest →
          </Link>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column: Input */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <h2 className="text-2xl font-bold text-white mb-6">Trade Scenario</h2>

            <div className="space-y-6">
              {/* Random Sample Button */}
              <button
                onClick={loadRandomScenario}
                className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 px-6 rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transition-all shadow-lg"
              >
                🎲 Load Random Sample
              </button>

              {/* Position Toggle */}
              <div>
                <label className="block text-white font-medium mb-3">Position</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setScenario({ ...scenario, position: 'long' })}
                    className={`py-3 px-4 rounded-xl font-semibold transition-all ${
                      scenario.position === 'long'
                        ? 'bg-green-500 text-white shadow-lg'
                        : 'bg-white/20 text-gray-300 hover:bg-white/30'
                    }`}
                  >
                    📈 Long
                  </button>
                  <button
                    onClick={() => setScenario({ ...scenario, position: 'short' })}
                    className={`py-3 px-4 rounded-xl font-semibold transition-all ${
                      scenario.position === 'short'
                        ? 'bg-red-500 text-white shadow-lg'
                        : 'bg-white/20 text-gray-300 hover:bg-white/30'
                    }`}
                  >
                    📉 Short
                  </button>
                </div>
              </div>

              {/* Leverage Slider */}
              <div>
                <label className="block text-white font-medium mb-3">
                  Leverage: <span className="text-purple-300">{scenario.leverage}x</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={scenario.leverage}
                  onChange={(e) =>
                    setScenario({ ...scenario, leverage: Number(e.target.value) })
                  }
                  className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
                <div className="flex justify-between text-sm text-gray-400 mt-1">
                  <span>1x</span>
                  <span>100x</span>
                </div>
              </div>

              {/* Collateral Input */}
              <div>
                <label className="block text-white font-medium mb-3">
                  Collateral (USD)
                </label>
                <input
                  type="number"
                  value={scenario.collateral}
                  onChange={(e) =>
                    setScenario({ ...scenario, collateral: Number(e.target.value) })
                  }
                  className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="Enter amount"
                />
              </div>

              {/* Duration Slider */}
              <div>
                <label className="block text-white font-medium mb-3">
                  Duration: <span className="text-purple-300">{duration} hours</span>
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

              {/* Calculate Button */}
              <button
                onClick={handleCalculate}
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-4 px-6 rounded-xl font-bold text-lg hover:from-blue-600 hover:to-cyan-600 transition-all shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? '⏳ Calculating...' : '🧮 Calculate Rate & Simulate'}
              </button>
            </div>

            {error && (
              <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="space-y-6">
            {/* Rate Results */}
            {rateData && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
                <h2 className="text-2xl font-bold text-white mb-6">💰 Lending Rate</h2>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 p-4 rounded-xl border border-blue-400/30">
                    <div className="text-sm text-blue-200 mb-1">Daily Rate</div>
                    <div className="text-3xl font-bold text-white">
                      {(rateData.lending_rate_per_day * 100).toFixed(4)}%
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 p-4 rounded-xl border border-purple-400/30">
                    <div className="text-sm text-purple-200 mb-1">Annual APR</div>
                    <div className="text-3xl font-bold text-white">
                      {(rateData.lending_rate_annualized * 100).toFixed(2)}%
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 p-4 rounded-xl border border-green-400/30">
                    <div className="text-sm text-green-200 mb-1">Base Rate</div>
                    <div className="text-xl font-bold text-white">
                      {(rateData.base_rate * 100).toFixed(4)}%
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-orange-500/20 to-red-500/20 p-4 rounded-xl border border-orange-400/30">
                    <div className="text-sm text-orange-200 mb-1">Vol Premium</div>
                    <div className="text-xl font-bold text-white">
                      {(rateData.volatility_premium * 100).toFixed(4)}%
                    </div>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-white/5 rounded-xl">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Predicted Vol (24h):</span>
                      <span className="text-white ml-2 font-semibold">
                        {rateData.predicted_volatility_24h.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Regime:</span>
                      <span className="text-white ml-2 font-semibold capitalize">
                        {rateData.volatility_regime.replace('_', ' ')}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Simulation Results */}
            {simulation && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
                <h2 className="text-2xl font-bold text-white mb-6">📊 Trade Simulation</h2>

                <div className="space-y-4">
                  <div className="bg-red-500/20 p-4 rounded-xl border border-red-400/30">
                    <div className="text-sm text-red-200 mb-2">Liquidation Risk</div>
                    <div className="flex items-baseline gap-2">
                      <div className="text-4xl font-bold text-white">
                        {(simulation.liquidation_probability * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-red-300">
                        ({simulation.liquidation_probability > 0.3 ? '⚠️ High Risk' : '✅ Moderate'})
                      </div>
                    </div>
                  </div>

                  <div className={`p-4 rounded-xl border ${
                    simulation.expected_pnl >= 0
                      ? 'bg-green-500/20 border-green-400/30'
                      : 'bg-red-500/20 border-red-400/30'
                  }`}>
                    <div className="text-sm text-gray-200 mb-2">Expected P&L</div>
                    <div className="text-3xl font-bold text-white">
                      ${simulation.expected_pnl.toFixed(2)}
                    </div>
                  </div>

                  <div className="bg-white/5 p-4 rounded-xl">
                    <div className="text-sm text-gray-400 mb-3">P&L Distribution</div>
                    <div className="space-y-2 text-sm">
                      {Object.entries(simulation.pnl_percentiles).map(([percentile, value]) => (
                        <div key={percentile} className="flex justify-between items-center">
                          <span className="text-gray-400">{percentile} percentile:</span>
                          <span className={`font-semibold ${
                            value >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            ${value.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-yellow-500/20 p-4 rounded-xl border border-yellow-400/30">
                    <div className="text-sm text-yellow-200 mb-2">Total Interest Cost</div>
                    <div className="text-2xl font-bold text-white">
                      ${simulation.total_interest_expected.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
