import React, { useEffect, useState } from 'react'
import { strategiesApi } from '../api/client'

/**
 * Renders editable inputs for every strategy-specific parameter returned by
 * GET /api/strategies.  Input types are inferred from the default value:
 *   boolean → checkbox
 *   number  → number input (step 1 for integers, 0.01 for floats)
 *   string  → text input
 *
 * Props:
 *   strategyName  string            — selected strategy class name
 *   config        object            — current param values (controlled)
 *   onChange      (config) => void  — called whenever any field changes
 */
function StrategyConfigForm({ strategyName, config, onChange }) {
  const [defaults, setDefaults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fetchError, setFetchError] = useState(null)

  useEffect(() => {
    if (!strategyName) {
      setDefaults(null)
      onChange({})
      return
    }

    setLoading(true)
    setFetchError(null)

    strategiesApi
      .list()
      .then((res) => {
        const match = res.data.find((s) => s.name === strategyName)
        const defaultConfig = match?.default_config ?? {}
        setDefaults(defaultConfig)
        onChange({ ...defaultConfig })
      })
      .catch(() => {
        setFetchError('Could not load strategy parameters.')
        setDefaults({})
        onChange({})
      })
      .finally(() => setLoading(false))
  }, [strategyName])

  if (!strategyName || defaults === null) return null
  if (loading) return <p style={{ color: '#6c757d' }}>Loading strategy parameters…</p>
  if (fetchError) return <p style={{ color: '#dc3545' }}>{fetchError}</p>
  if (Object.keys(defaults).length === 0) return null

  const numericAndString = Object.entries(defaults).filter(([, v]) => typeof v !== 'boolean')
  const booleans = Object.entries(defaults).filter(([, v]) => typeof v === 'boolean')

  return (
    <div
      className="card"
      style={{ marginBottom: '20px', background: '#f8f9fa', border: '1px solid #dee2e6' }}
    >
      <h3 style={{ marginBottom: '16px' }}>{strategyName} Parameters</h3>

      {numericAndString.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          {numericAndString.map(([key, defaultValue]) => (
            <div className="form-group" key={key}>
              <label htmlFor={`scf-${key}`}>{toLabel(key)}</label>
              <input
                id={`scf-${key}`}
                type={inferInputType(defaultValue)}
                name={key}
                value={config[key] ?? defaultValue}
                onChange={(e) => handleChange(e, defaults, config, onChange)}
                step={typeof defaultValue === 'number' ? inferStep(defaultValue) : undefined}
              />
              <small style={{ color: '#6c757d' }}>default: {String(defaultValue)}</small>
            </div>
          ))}
        </div>
      )}

      {booleans.length > 0 && (
        <div
          style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '12px' }}
        >
          {booleans.map(([key, defaultValue]) => (
            <div className="form-group" key={key}>
              <label
                htmlFor={`scf-${key}`}
                style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
              >
                <input
                  id={`scf-${key}`}
                  type="checkbox"
                  name={key}
                  checked={config[key] ?? defaultValue}
                  onChange={(e) => handleChange(e, defaults, config, onChange)}
                />
                {toLabel(key)}
              </label>
              <small style={{ color: '#6c757d' }}>default: {String(defaultValue)}</small>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pure helpers (exported for testing)
// ---------------------------------------------------------------------------

export function toLabel(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export function inferInputType(value) {
  if (typeof value === 'boolean') return 'checkbox'
  if (typeof value === 'number') return 'number'
  return 'text'
}

export function inferStep(value) {
  return Number.isInteger(value) ? '1' : '0.01'
}

function handleChange(e, defaults, config, onChange) {
  const { name, value, type, checked } = e.target
  let parsed
  if (type === 'checkbox') {
    parsed = checked
  } else if (type === 'number') {
    parsed = Number.isInteger(defaults[name]) ? parseInt(value, 10) : parseFloat(value)
  } else {
    parsed = value
  }
  onChange({ ...config, [name]: parsed })
}

export default StrategyConfigForm
