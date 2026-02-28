import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { statsApi, operationsApi } from '../api/client'
import { formatCurrency, formatPercent } from '../utils/formatters'

function Dashboard() {
  const [overallStats, setOverallStats] = useState(null)
  const [recentOperations, setRecentOperations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      const [statsRes, opsRes] = await Promise.all([
        statsApi.overall(),
        operationsApi.list('active'),
      ])
      setOverallStats(statsRes.data)
      setRecentOperations(opsRes.data.slice(0, 5))
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading && !overallStats) {
    return <div className="loading">Loading dashboard...</div>
  }

  if (error) {
    return <div className="error">Error: {error}</div>
  }

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <Link to="/operations/create" className="btn btn-primary">
          Create Operation
        </Link>
      </div>

      {overallStats && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Total Operations</div>
            <div className="stat-value">{overallStats.total_operations}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Active Operations</div>
            <div className="stat-value">{overallStats.active_operations}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Total Trades</div>
            <div className="stat-value">{overallStats.total_trades}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Balance {overallStats.currency && `(${overallStats.currency})`}</div>
            <div className="stat-value">{formatCurrency(overallStats.balance)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Equity</div>
            <div className="stat-value">{formatCurrency(overallStats.equity)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Floating P/L</div>
            <div className={`stat-value ${overallStats.floating_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`}>
              {formatCurrency(overallStats.floating_pnl)}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Margin Used</div>
            <div className="stat-value">{formatCurrency(overallStats.margin_used)}</div>
          </div>
        </div>
      )}

      <div className="card">
        <h2>Recent Active Operations</h2>
        {recentOperations.length === 0 ? (
          <p>No active operations</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Strategy</th>
                <th>Status</th>
                <th>P/L</th>
                <th>P/L %</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {recentOperations.map((op) => (
                <tr key={op.id}>
                  <td>{op.asset}</td>
                  <td>{op.strategy_name}</td>
                  <td>
                    <span className={`status-badge status-${op.status}`}>
                      {op.status}
                    </span>
                  </td>
                  <td className={op.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    {formatCurrency(op.total_pnl)}
                  </td>
                  <td className={op.total_pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    {formatPercent(op.total_pnl_pct)}
                  </td>
                  <td>
                    <Link to={`/operations/${op.id}`} className="btn btn-secondary">
                      View
                    </Link>
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

export default Dashboard

