import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { operationsApi } from '../api/client'
import { formatCurrency, formatPercent } from '../utils/formatters'

const STATUS_TABS = ['', 'active', 'paused', 'closed']
const TAB_LABELS = { '': 'All', active: 'Active', paused: 'Paused', closed: 'Closed' }

function Operations() {
  const [operations, setOperations] = useState([])
  const [allOperations, setAllOperations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [statusFilter, setStatusFilter] = useState('')

  useEffect(() => {
    loadOperations()
    const interval = setInterval(loadOperations, 5000)
    return () => clearInterval(interval)
  }, [statusFilter])

  // Load all operations once for count badges
  useEffect(() => {
    operationsApi.list(undefined)
      .then((res) => setAllOperations(res.data))
      .catch(() => {})
  }, [])

  const loadOperations = async () => {
    try {
      setLoading(true)
      const res = await operationsApi.list(statusFilter || undefined)
      setOperations(res.data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handlePause = async (id) => {
    try {
      await operationsApi.pause(id)
      loadOperations()
    } catch (err) {
      alert(`Error pausing operation: ${err.message}`)
    }
  }

  const handleResume = async (id) => {
    try {
      await operationsApi.resume(id)
      loadOperations()
    } catch (err) {
      alert(`Error resuming operation: ${err.message}`)
    }
  }

  const handleStop = async (id) => {
    if (!window.confirm('Are you sure you want to stop this operation?')) return
    try {
      await operationsApi.delete(id)
      loadOperations()
    } catch (err) {
      alert(`Error stopping operation: ${err.message}`)
    }
  }

  const countFor = (status) => {
    if (!status) return allOperations.length
    return allOperations.filter((op) => op.status === status).length
  }

  if (loading && operations.length === 0) {
    return <div className="loading">Loading operations...</div>
  }

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Trading Operations</h1>
        <Link to="/operations/create" className="btn btn-primary">
          + New Operation
        </Link>
      </div>

      {error && <div className="error">Error: {error}</div>}

      <div className="card">
        {/* Tab strip filter */}
        <div style={{ display: 'flex', gap: '4px', marginBottom: '20px', borderBottom: '1px solid var(--border)', paddingBottom: '0' }}>
          {STATUS_TABS.map((status) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              style={{
                padding: '8px 16px',
                background: 'none',
                border: 'none',
                borderBottom: statusFilter === status ? '2px solid var(--accent)' : '2px solid transparent',
                color: statusFilter === status ? 'var(--text-primary)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: statusFilter === status ? '600' : '400',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'color 0.15s',
                marginBottom: '-1px',
              }}
            >
              {TAB_LABELS[status]}
              <span style={{
                background: 'var(--bg-elevated)',
                color: 'var(--text-secondary)',
                fontSize: '11px',
                padding: '1px 6px',
                borderRadius: '10px',
                fontWeight: '400',
              }}>
                {countFor(status)}
              </span>
            </button>
          ))}
        </div>

        {operations.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>No operations found</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Strategy</th>
                <th>Bar Sizes</th>
                <th>Status</th>
                <th>Initial Capital</th>
                <th>Current Capital</th>
                <th>P/L</th>
                <th>P/L %</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {operations.map((op) => (
                <tr key={op.id}>
                  <td style={{ fontWeight: '600' }}>{op.asset}</td>
                  <td>{op.strategy_name}</td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>{op.bar_sizes.join(', ')}</td>
                  <td>
                    <span className={`status-badge status-${op.status}`}>{op.status}</span>
                  </td>
                  <td>{formatCurrency(op.initial_capital)}</td>
                  <td>{formatCurrency(op.current_capital)}</td>
                  <td className={op.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    {formatCurrency(op.total_pnl)}
                  </td>
                  <td className={op.total_pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    {formatPercent(op.total_pnl_pct)}
                  </td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
                    {new Date(op.created_at).toLocaleDateString()}
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <Link to={`/operations/${op.id}`} className="btn btn-secondary">View</Link>
                      {op.status === 'active' && (
                        <button onClick={() => handlePause(op.id)} className="btn btn-secondary" title="Pause">⏸</button>
                      )}
                      {op.status === 'paused' && (
                        <button onClick={() => handleResume(op.id)} className="btn btn-success" title="Resume">▶</button>
                      )}
                      {(op.status === 'active' || op.status === 'paused') && (
                        <button onClick={() => handleStop(op.id)} className="btn btn-danger" title="Stop">■</button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export default Operations

