import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { operationsApi, strategiesApi } from '../api/client'

// Default IFS-specific parameters (mirrors InstitutionalFlowStrategy.__init__ defaults)
const IFS_DEFAULTS = {
  hurst_threshold: 0.52,
  use_mean_reversion_mode: false,
  use_rsi_filter: true,
  rsi_long_max: 55.0,
  rsi_short_min: 45.0,
  use_adx_filter: true,
  adx_min: 25.0,
  min_fvg_atr_ratio: 0.5,
  min_rr_ratio: 2.0,
  cooldown_bars: 8,
  session_start_hour: 7,
  session_end_hour: 17,
}

function CreateOperation() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [strategies, setStrategies] = useState([])
  const [ifsConfig, setIfsConfig] = useState({ ...IFS_DEFAULTS })

  const [formData, setFormData] = useState({
    asset: '',
    bar_sizes: [],
    primary_bar_size: '',
    strategy_name: '',
    strategy_config: {},
    initial_capital: 10000,
    stop_loss_type: 'ATR',
    stop_loss_value: 1.5,
    take_profit_type: 'RISK_REWARD',
    take_profit_value: 2.0,
    crash_recovery_mode: 'CLOSE_ALL',
    emergency_stop_loss_pct: 0.05,
    data_retention_bars: 1000,
  })

  useEffect(() => {
    strategiesApi.list()
      .then((res) => setStrategies(res.data.map((s) => s.name)))
      .catch(() => {
        // Fallback to known strategies if API is unavailable
        setStrategies([
          'AdaptiveMultiIndicatorStrategy',
          'ATRBreakout',
          'BuyAndHoldStrategy',
          'HammerShootingStar',
          'InstitutionalFlowStrategy',
          'MARSIStrategy',
          'MarketStructureStrategy',
          'MeanReversionStrategy',
          'MomentumStrategy',
          'MultiTimeframeStrategy',
          'PatternStrategy',
          'PatternTriangleStrategy',
          'RSIStrategy',
          'TriangleStrategy',
        ])
      })
  }, [])

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    if (type === 'checkbox') {
      if (name === 'bar_size') {
        const barSizes = formData.bar_sizes.includes(value)
          ? formData.bar_sizes.filter((bs) => bs !== value)
          : [...formData.bar_sizes, value]
        setFormData({ ...formData, bar_sizes: barSizes })
      }
    } else {
      setFormData({ ...formData, [name]: value })
    }
  }

  const handleIfsChange = (e) => {
    const { name, value, type, checked } = e.target
    setIfsConfig({
      ...ifsConfig,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? parseFloat(value) : value),
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const strategy_config =
        formData.strategy_name === 'InstitutionalFlowStrategy'
          ? { ...ifsConfig }
          : {}

      const payload = {
        ...formData,
        strategy_config,
        initial_capital: parseFloat(formData.initial_capital),
        stop_loss_value: parseFloat(formData.stop_loss_value),
        take_profit_value: parseFloat(formData.take_profit_value),
        emergency_stop_loss_pct: parseFloat(formData.emergency_stop_loss_pct),
        data_retention_bars: parseInt(formData.data_retention_bars),
      }

      const res = await operationsApi.create(payload)
      navigate(`/operations/${res.data.id}`)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const availableBarSizes = [
    '5 mins',
    '15 mins',
    '30 mins',
    '1 hour',
    '4 hours',
    '1 day',
    '1 week',
  ]

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Create Trading Operation</h1>
      </div>

      {error && <div className="error">Error: {error}</div>}

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Asset Symbol *</label>
            <input
              type="text"
              name="asset"
              value={formData.asset}
              onChange={handleChange}
              placeholder="e.g., USD-CAD"
              required
            />
          </div>

          <div className="form-group">
            <label>Bar Sizes *</label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
              {availableBarSizes.map((bs) => (
                <label key={bs} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <input
                    type="checkbox"
                    name="bar_size"
                    value={bs}
                    checked={formData.bar_sizes.includes(bs)}
                    onChange={handleChange}
                  />
                  {bs}
                </label>
              ))}
            </div>
          </div>

          <div className="form-group">
            <label>Primary Bar Size *</label>
            <select
              name="primary_bar_size"
              value={formData.primary_bar_size}
              onChange={handleChange}
              required
            >
              <option value="">Select primary bar size</option>
              {formData.bar_sizes.map((bs) => (
                <option key={bs} value={bs}>
                  {bs}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Strategy *</label>
            <select
              name="strategy_name"
              value={formData.strategy_name}
              onChange={handleChange}
              required
            >
              <option value="">Select strategy</option>
              {strategies.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          {formData.strategy_name === 'InstitutionalFlowStrategy' && (
            <div className="card" style={{ marginBottom: '20px', background: '#f8f9fa', border: '1px solid #dee2e6' }}>
              <h3 style={{ marginBottom: '20px' }}>InstitutionalFlowStrategy Parameters</h3>
              <p style={{ marginBottom: '16px', color: '#6c757d', fontSize: '14px' }}>
                Regime-switching FVG strategy. Defaults reflect the tuned configuration
                from backtest analysis (trend-only mode, RSI + ADX confirmation).
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div className="form-group">
                  <label>Hurst Threshold</label>
                  <input type="number" name="hurst_threshold" value={ifsConfig.hurst_threshold}
                    onChange={handleIfsChange} min="0" max="1" step="0.01" />
                  <small style={{ color: '#6c757d' }}>H &gt; threshold = trending regime (default 0.52)</small>
                </div>

                <div className="form-group">
                  <label>Min FVG ATR Ratio</label>
                  <input type="number" name="min_fvg_atr_ratio" value={ifsConfig.min_fvg_atr_ratio}
                    onChange={handleIfsChange} min="0" step="0.05" />
                  <small style={{ color: '#6c757d' }}>Min gap size relative to ATR (default 0.5)</small>
                </div>

                <div className="form-group">
                  <label>Min R:R Ratio</label>
                  <input type="number" name="min_rr_ratio" value={ifsConfig.min_rr_ratio}
                    onChange={handleIfsChange} min="0.5" step="0.1" />
                  <small style={{ color: '#6c757d' }}>Reject setups below this reward/risk (default 2.0)</small>
                </div>

                <div className="form-group">
                  <label>Cooldown Bars</label>
                  <input type="number" name="cooldown_bars" value={ifsConfig.cooldown_bars}
                    onChange={handleIfsChange} min="1" step="1" />
                  <small style={{ color: '#6c757d' }}>Minimum bars between signals (default 8)</small>
                </div>

                <div className="form-group">
                  <label>Session Start Hour (UTC)</label>
                  <input type="number" name="session_start_hour" value={ifsConfig.session_start_hour}
                    onChange={handleIfsChange} min="0" max="23" step="1" />
                </div>

                <div className="form-group">
                  <label>Session End Hour (UTC)</label>
                  <input type="number" name="session_end_hour" value={ifsConfig.session_end_hour}
                    onChange={handleIfsChange} min="0" max="23" step="1" />
                  <small style={{ color: '#6c757d' }}>Default 7–17 UTC covers London + NY sessions</small>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '8px' }}>
                <div className="form-group">
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" name="use_rsi_filter" checked={ifsConfig.use_rsi_filter}
                      onChange={handleIfsChange} />
                    Enable RSI Filter
                  </label>
                  <small style={{ color: '#6c757d' }}>Require RSI confirmation before entry</small>
                </div>

                <div className="form-group">
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" name="use_adx_filter" checked={ifsConfig.use_adx_filter}
                      onChange={handleIfsChange} />
                    Enable ADX Filter
                  </label>
                  {ifsConfig.use_adx_filter && (
                    <input type="number" name="adx_min" value={ifsConfig.adx_min}
                      onChange={handleIfsChange} min="0" step="1"
                      placeholder="Min ADX" style={{ marginTop: '6px' }} />
                  )}
                  <small style={{ color: '#6c757d' }}>Require minimum trend strength (default ADX ≥ 25)</small>
                </div>

                <div className="form-group">
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" name="use_mean_reversion_mode" checked={ifsConfig.use_mean_reversion_mode}
                      onChange={handleIfsChange} />
                    Enable Mean-Reversion Mode
                  </label>
                  <small style={{ color: '#6c757d' }}>Fades FVG gaps in ranging markets. Off by default (loses on trending pairs)</small>
                </div>
              </div>
            </div>
          )}

          <div className="form-group">
            <label>Initial Capital *</label>
            <input
              type="number"
              name="initial_capital"
              value={formData.initial_capital}
              onChange={handleChange}
              min="0"
              step="0.01"
              required
            />
          </div>

          <h3 style={{ marginTop: '30px', marginBottom: '20px' }}>Risk Management</h3>

          <div className="form-group">
            <label>Stop Loss Type</label>
            <select
              name="stop_loss_type"
              value={formData.stop_loss_type}
              onChange={handleChange}
            >
              <option value="ATR">ATR</option>
              <option value="PERCENTAGE">Percentage</option>
              <option value="FIXED">Fixed</option>
            </select>
          </div>

          <div className="form-group">
            <label>Stop Loss Value</label>
            <input
              type="number"
              name="stop_loss_value"
              value={formData.stop_loss_value}
              onChange={handleChange}
              min="0"
              step="0.01"
            />
          </div>

          <div className="form-group">
            <label>Take Profit Type</label>
            <select
              name="take_profit_type"
              value={formData.take_profit_type}
              onChange={handleChange}
            >
              <option value="RISK_REWARD">Risk-Reward Ratio</option>
              <option value="ATR">ATR</option>
              <option value="PERCENTAGE">Percentage</option>
              <option value="FIXED">Fixed</option>
            </select>
          </div>

          <div className="form-group">
            <label>Take Profit Value</label>
            <input
              type="number"
              name="take_profit_value"
              value={formData.take_profit_value}
              onChange={handleChange}
              min="0"
              step="0.01"
            />
          </div>

          <h3 style={{ marginTop: '30px', marginBottom: '20px' }}>Crash Recovery</h3>

          <div className="form-group">
            <label>Crash Recovery Mode</label>
            <select
              name="crash_recovery_mode"
              value={formData.crash_recovery_mode}
              onChange={handleChange}
            >
              <option value="CLOSE_ALL">Close All (Safest)</option>
              <option value="RESUME">Resume</option>
              <option value="EMERGENCY_EXIT">Emergency Exit</option>
            </select>
          </div>

          <div className="form-group">
            <label>Emergency Stop Loss %</label>
            <input
              type="number"
              name="emergency_stop_loss_pct"
              value={formData.emergency_stop_loss_pct}
              onChange={handleChange}
              min="0"
              max="1"
              step="0.01"
            />
          </div>

          <div className="form-group">
            <label>Data Retention Bars</label>
            <input
              type="number"
              name="data_retention_bars"
              value={formData.data_retention_bars}
              onChange={handleChange}
              min="100"
              step="100"
            />
          </div>

          <div style={{ marginTop: '30px', display: 'flex', gap: '10px' }}>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Creating...' : 'Create Operation'}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => navigate('/operations')}
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default CreateOperation

