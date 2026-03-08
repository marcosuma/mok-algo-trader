import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { operationsApi, strategiesApi } from '../api/client'
import StrategyConfigForm from '../components/StrategyConfigForm'

function CreateOperation() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [strategies, setStrategies] = useState([])
  const [strategyConfig, setStrategyConfig] = useState({})

  const [formData, setFormData] = useState({
    asset: '',
    bar_sizes: [],
    primary_bar_size: '',
    strategy_name: '',
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
    if (type === 'checkbox' && name === 'bar_size') {
      const barSizes = formData.bar_sizes.includes(value)
        ? formData.bar_sizes.filter((bs) => bs !== value)
        : [...formData.bar_sizes, value]
      setFormData({ ...formData, bar_sizes: barSizes })
    } else {
      setFormData({ ...formData, [name]: value })
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const payload = {
        ...formData,
        strategy_config: strategyConfig,
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

  const availableBarSizes = ['1 min', '5 mins', '15 mins', '30 mins', '1 hour', '4 hours', '1 day', '1 week']

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
                <option key={bs} value={bs}>{bs}</option>
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
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          <StrategyConfigForm
            strategyName={formData.strategy_name}
            config={strategyConfig}
            onChange={setStrategyConfig}
          />

          <h3 style={{ marginTop: '30px', marginBottom: '20px' }}>Risk Management</h3>

          <div className="form-group">
            <label>Stop Loss Type</label>
            <select name="stop_loss_type" value={formData.stop_loss_type} onChange={handleChange}>
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
            <select name="take_profit_type" value={formData.take_profit_type} onChange={handleChange}>
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
            <select name="crash_recovery_mode" value={formData.crash_recovery_mode} onChange={handleChange}>
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
