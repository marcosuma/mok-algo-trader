import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  operationsApi,
  positionsApi,
  tradesApi,
  ordersApi,
  statsApi,
  marketDataApi,
} from '../api/client'
import { formatCurrency, formatPercent, formatDate, formatForexPrice } from '../utils/formatters'
import MarketDataChart from '../components/MarketDataChart'

// Parse bar size string to milliseconds (with a small buffer for data availability)
const parseBarSizeToMs = (barSize) => {
  if (!barSize) return null

  const match = barSize.match(/^(\d+)\s*(mins?|hours?|days?|weeks?)$/i)
  if (!match) return null

  const value = parseInt(match[1], 10)
  const unit = match[2].toLowerCase()

  let ms = 0
  if (unit.startsWith('min')) {
    ms = value * 60 * 1000
  } else if (unit.startsWith('hour')) {
    ms = value * 60 * 60 * 1000
  } else if (unit.startsWith('day')) {
    ms = value * 24 * 60 * 60 * 1000
  } else if (unit.startsWith('week')) {
    ms = value * 7 * 24 * 60 * 60 * 1000
  }

  // Add a 10-second buffer to ensure new data is available
  return ms + 10000
}

// Format milliseconds to human readable countdown
const formatCountdown = (ms) => {
  if (ms <= 0) return 'Refreshing...'

  const seconds = Math.floor(ms / 1000) % 60
  const minutes = Math.floor(ms / 60000) % 60
  const hours = Math.floor(ms / 3600000)

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`
  } else if (minutes > 0) {
    return `${minutes}m ${seconds}s`
  }
  return `${seconds}s`
}

function OperationDetail() {
  const { id } = useParams()
  const [operation, setOperation] = useState(null)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState([])
  const [orders, setOrders] = useState([])
  const [stats, setStats] = useState(null)
  const [marketData, setMarketData] = useState([])
  const [marketDataCount, setMarketDataCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [marketDataLoading, setMarketDataLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedBarSize, setSelectedBarSize] = useState(null)
  const [selectedIndicators, setSelectedIndicators] = useState([])
  const [availableIndicators, setAvailableIndicators] = useState([])

  // Live refresh state
  const [liveRefresh, setLiveRefresh] = useState(false)
  const [nextRefreshIn, setNextRefreshIn] = useState(null)
  const [lastRefreshTime, setLastRefreshTime] = useState(null)

  useEffect(() => {
    loadData()
    loadMarketDataCount()
    const interval = setInterval(() => {
      loadData()
      loadMarketDataCount()
    }, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [id])

  const loadData = async () => {
    try {
      setLoading(true)
      const [
        opRes,
        posRes,
        tradesRes,
        ordersRes,
        statsRes,
      ] = await Promise.all([
        operationsApi.get(id),
        positionsApi.list(id),
        tradesApi.list(id),
        ordersApi.list(id),
        statsApi.operation(id),
      ])
      setOperation(opRes.data)
      setPositions(posRes.data)
      setTrades(tradesRes.data)
      setOrders(ordersRes.data)
      setStats(statsRes.data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Load market data count only (for tab display)
  const loadMarketDataCount = async () => {
    try {
      const response = await marketDataApi.count(id)
      setMarketDataCount(response.data.count)
    } catch (err) {
      console.error('Error loading market data count:', err)
    }
  }

  const loadMarketData = async (barSize = null) => {
    try {
      setMarketDataLoading(true)
      // Request all available data (use a large limit or no limit)
      const response = await marketDataApi.list(id, barSize, 100000) // Get up to 100k bars
      const data = response.data
      setMarketData(data)

      // Update last refresh time for live refresh countdown
      setLastRefreshTime(Date.now())

      // Extract available indicators from the data
      if (data.length > 0) {
        const indicators = new Set()
        data.forEach((item) => {
          if (item.indicators) {
            Object.keys(item.indicators).forEach((ind) => indicators.add(ind))
          }
        })
        setAvailableIndicators(Array.from(indicators).sort())

        // Set default bar size to primary bar size if not set
        if (!selectedBarSize && operation) {
          setSelectedBarSize(operation.primary_bar_size)
        }
      }
    } catch (err) {
      console.error('Error loading market data:', err)
    } finally {
      setMarketDataLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'market-data' && operation) {
      // Set default bar size if not set
      const barSizeToUse = selectedBarSize || operation.primary_bar_size
      if (!selectedBarSize) {
        setSelectedBarSize(operation.primary_bar_size)
      }
      loadMarketData(barSizeToUse)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, selectedBarSize, id])

  // Live refresh effect
  useEffect(() => {
    if (!liveRefresh || activeTab !== 'market-data' || !selectedBarSize) {
      setNextRefreshIn(null)
      return
    }

    const refreshInterval = parseBarSizeToMs(selectedBarSize)
    if (!refreshInterval) {
      console.warn(`Could not parse bar size: ${selectedBarSize}`)
      return
    }

    // Calculate time until next refresh based on last refresh
    const calculateTimeUntilRefresh = () => {
      if (!lastRefreshTime) return refreshInterval
      const elapsed = Date.now() - lastRefreshTime
      return Math.max(0, refreshInterval - elapsed)
    }

    // Countdown timer - updates every second
    const countdownInterval = setInterval(() => {
      const timeLeft = calculateTimeUntilRefresh()
      setNextRefreshIn(timeLeft)

      // Trigger refresh when countdown reaches 0
      if (timeLeft <= 0) {
        loadMarketData(selectedBarSize)
      }
    }, 1000)

    // Initial countdown
    setNextRefreshIn(calculateTimeUntilRefresh())

    return () => {
      clearInterval(countdownInterval)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveRefresh, selectedBarSize, activeTab, lastRefreshTime])

  // Reset live refresh when bar size changes
  useEffect(() => {
    if (liveRefresh && selectedBarSize) {
      setLastRefreshTime(Date.now())
    }
  }, [selectedBarSize, liveRefresh])

  if (loading && !operation) {
    return <div className="loading">Loading operation details...</div>
  }

  if (error || !operation) {
    return <div className="error">Error: {error || 'Operation not found'}</div>
  }

  const openPositions = positions.filter((p) => p.status === 'OPEN')
  const closedPositions = positions.filter((p) => p.status === 'CLOSED')

  return (
    <div className="container">
      <div className="page-header">
        <div>
          <Link to="/operations" style={{ color: '#6c757d', textDecoration: 'none' }}>
            ← Back to Operations
          </Link>
          <h1 className="page-title" style={{ marginTop: '10px' }}>
            {operation.asset} - {operation.strategy_name}
          </h1>
        </div>
        <div>
          <span className={`status-badge status-${operation.status}`}>
            {operation.status}
          </span>
        </div>
      </div>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Total P/L</div>
            <div className={`stat-value ${stats.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`}>
              {formatCurrency(stats.total_pnl)}
            </div>
            <div className="stat-label">{formatPercent(stats.total_pnl_pct)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Total Trades</div>
            <div className="stat-value">{stats.total_trades}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Winning Trades</div>
            <div className="stat-value">{stats.winning_trades}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Losing Trades</div>
            <div className="stat-value">{stats.losing_trades}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Open Positions</div>
            <div className="stat-value">{stats.open_positions}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Current Capital</div>
            <div className="stat-value">{formatCurrency(stats.current_capital)}</div>
          </div>
        </div>
      )}

      <div className="card">
        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', borderBottom: '1px solid #ddd', paddingBottom: '10px' }}>
          <button
            className={`btn ${activeTab === 'overview' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`btn ${activeTab === 'positions' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab('positions')}
          >
            Positions ({positions.length})
          </button>
          <button
            className={`btn ${activeTab === 'trades' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab('trades')}
          >
            Trades ({trades.length})
          </button>
          <button
            className={`btn ${activeTab === 'orders' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab('orders')}
          >
            Orders ({orders.length})
          </button>
          <button
            className={`btn ${activeTab === 'market-data' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab('market-data')}
          >
            Market Data ({marketDataCount})
          </button>
        </div>

        {activeTab === 'overview' && (
          <div>
            <h3>Operation Details</h3>
            <table>
              <tbody>
                <tr>
                  <td><strong>Asset</strong></td>
                  <td>{operation.asset}</td>
                </tr>
                <tr>
                  <td><strong>Strategy</strong></td>
                  <td>{operation.strategy_name}</td>
                </tr>
                <tr>
                  <td><strong>Bar Sizes</strong></td>
                  <td>{operation.bar_sizes.join(', ')}</td>
                </tr>
                <tr>
                  <td><strong>Primary Bar Size</strong></td>
                  <td>{operation.primary_bar_size}</td>
                </tr>
                <tr>
                  <td><strong>Status</strong></td>
                  <td>
                    <span className={`status-badge status-${operation.status}`}>
                      {operation.status}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td><strong>Initial Capital</strong></td>
                  <td>{formatCurrency(operation.initial_capital)}</td>
                </tr>
                <tr>
                  <td><strong>Current Capital</strong></td>
                  <td>{formatCurrency(operation.current_capital)}</td>
                </tr>
                <tr>
                  <td><strong>Total P/L</strong></td>
                  <td className={operation.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    {formatCurrency(operation.total_pnl)} ({formatPercent(operation.total_pnl_pct)})
                  </td>
                </tr>
                <tr>
                  <td><strong>Created</strong></td>
                  <td>{formatDate(operation.created_at)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'positions' && (
          <div>
            <h3>Open Positions</h3>
            {openPositions.length === 0 ? (
              <p>No open positions</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Quantity</th>
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>Unrealized P/L</th>
                    <th>Unrealized P/L %</th>
                    <th>Stop Loss</th>
                    <th>Take Profit</th>
                  </tr>
                </thead>
                <tbody>
                  {openPositions.map((pos) => (
                    <tr key={pos.id}>
                      <td>{pos.symbol}</td>
                      <td>{pos.side}</td>
                      <td>{pos.quantity}</td>
                      <td>{formatForexPrice(pos.entry_price)}</td>
                      <td>{formatForexPrice(pos.current_price)}</td>
                      <td className={pos.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                        {formatCurrency(pos.unrealized_pnl)}
                      </td>
                      <td className={pos.unrealized_pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                        {formatPercent(pos.unrealized_pnl_pct)}
                      </td>
                      <td>{pos.stop_loss ? formatForexPrice(pos.stop_loss) : '-'}</td>
                      <td>{pos.take_profit ? formatForexPrice(pos.take_profit) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {closedPositions.length > 0 && (
              <>
                <h3 style={{ marginTop: '30px' }}>Closed Positions</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th>Quantity</th>
                      <th>Entry Price</th>
                      <th>Close Price</th>
                      <th>P/L</th>
                      <th>Reason</th>
                      <th>Opened</th>
                      <th>Closed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {closedPositions.map((pos) => (
                      <tr key={pos.id}>
                        <td>{pos.symbol}</td>
                        <td>{pos.side}</td>
                        <td>{pos.quantity}</td>
                        <td>{formatForexPrice(pos.entry_price)}</td>
                        <td>{pos.close_price ? formatForexPrice(pos.close_price) : '-'}</td>
                        <td className={(pos.realized_pnl || 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                          {formatCurrency(pos.realized_pnl || 0)} ({formatPercent(pos.realized_pnl_pct || 0)})
                        </td>
                        <td>{pos.close_reason || '-'}</td>
                        <td>{formatDate(pos.opened_at)}</td>
                        <td>{formatDate(pos.closed_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>
        )}

        {activeTab === 'trades' && (
          <div>
            <h3>Completed Trades</h3>
            {trades.length === 0 ? (
              <p>No completed trades</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Entry Price</th>
                    <th>Close Price</th>
                    <th>Quantity</th>
                    <th>P/L</th>
                    <th>P/L %</th>
                    <th>Reason</th>
                    <th>Commission</th>
                    <th>Duration</th>
                    <th>Opened</th>
                    <th>Closed</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((trade) => (
                    <tr key={trade.id}>
                      <td>{trade.symbol}</td>
                      <td>{trade.side}</td>
                      <td>{formatForexPrice(trade.entry_price)}</td>
                      <td>{trade.close_price ? formatForexPrice(trade.close_price) : '-'}</td>
                      <td>{trade.quantity}</td>
                      <td className={(trade.realized_pnl || 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                        {formatCurrency(trade.realized_pnl || 0)}
                      </td>
                      <td className={(trade.realized_pnl_pct || 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                        {formatPercent(trade.realized_pnl_pct || 0)}
                      </td>
                      <td>{trade.close_reason || '-'}</td>
                      <td>{formatCurrency(trade.total_commission || 0)}</td>
                      <td>{trade.duration_seconds ? (trade.duration_seconds / 3600).toFixed(2) + ' hours' : '-'}</td>
                      <td>{formatDate(trade.opened_at)}</td>
                      <td>{formatDate(trade.closed_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {activeTab === 'orders' && (
          <div>
            <h3>Orders</h3>
            {orders.length === 0 ? (
              <p>No orders</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Type</th>
                    <th>Intent</th>
                    <th>Qty</th>
                    <th>Price</th>
                    <th>Status</th>
                    <th>Filled Qty</th>
                    <th>Avg Fill Price</th>
                    <th>Source</th>
                    <th>Created</th>
                    <th>Filled</th>
                  </tr>
                </thead>
                <tbody>
                  {orders.map((order) => (
                    <tr key={order.id}>
                      <td>{order.symbol}</td>
                      <td>{order.side}</td>
                      <td>{order.order_type}</td>
                      <td>{order.intent}</td>
                      <td>{order.requested_quantity}</td>
                      <td>{order.requested_price ? formatForexPrice(order.requested_price) : 'MARKET'}</td>
                      <td>
                        <span className={`status-badge status-${order.status.toLowerCase()}`}>
                          {order.status}
                        </span>
                      </td>
                      <td>{order.filled_quantity}</td>
                      <td>{order.avg_fill_price ? formatForexPrice(order.avg_fill_price) : '-'}</td>
                      <td>{order.source}</td>
                      <td>{formatDate(order.created_at)}</td>
                      <td>{order.filled_at ? formatDate(order.filled_at) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}

        {activeTab === 'market-data' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '15px' }}>
              <h3 style={{ margin: 0 }}>Market Data</h3>
              <div style={{ display: 'flex', gap: '15px', alignItems: 'center', flexWrap: 'wrap' }}>
                {operation && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <label htmlFor="bar-size-select" style={{ fontWeight: '600' }}>Bar Size:</label>
                    <select
                      id="bar-size-select"
                      value={selectedBarSize || ''}
                      onChange={(e) => setSelectedBarSize(e.target.value)}
                      style={{
                        padding: '8px 12px',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        fontSize: '14px',
                      }}
                    >
                      {operation.bar_sizes.map((bs) => (
                        <option key={bs} value={bs}>
                          {bs}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                {availableIndicators.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <label htmlFor="indicator-select" style={{ fontWeight: '600' }}>Indicators:</label>
                    <select
                      id="indicator-select"
                      multiple
                      value={selectedIndicators}
                      onChange={(e) => {
                        const values = Array.from(e.target.selectedOptions, (option) => option.value)
                        setSelectedIndicators(values)
                      }}
                      style={{
                        padding: '8px 12px',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        fontSize: '14px',
                        minWidth: '200px',
                      }}
                    >
                      {availableIndicators.map((ind) => (
                        <option key={ind} value={ind}>
                          {ind}
                        </option>
                      ))}
                    </select>
                    <small style={{ color: '#6c757d' }}>(Hold Ctrl/Cmd to select multiple)</small>
                  </div>
                )}

                {/* Live Refresh Controls */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '8px 16px',
                  backgroundColor: liveRefresh ? '#e8f5e9' : '#f5f5f5',
                  borderRadius: '8px',
                  border: liveRefresh ? '1px solid #4caf50' : '1px solid #ddd',
                  transition: 'all 0.3s ease'
                }}>
                  {/* Manual Refresh Button */}
                  <button
                    onClick={() => loadMarketData(selectedBarSize)}
                    disabled={marketDataLoading}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      padding: '6px 10px',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: marketDataLoading ? 'not-allowed' : 'pointer',
                      fontWeight: '500',
                      fontSize: '13px',
                      backgroundColor: '#2196F3',
                      color: '#fff',
                      opacity: marketDataLoading ? 0.6 : 1,
                      transition: 'opacity 0.2s ease'
                    }}
                    title="Refresh now"
                  >
                    {marketDataLoading ? '⏳' : '🔄'}
                  </button>

                  {/* Live Toggle Button */}
                  <button
                    onClick={() => setLiveRefresh(!liveRefresh)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '6px 12px',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontWeight: '600',
                      fontSize: '13px',
                      backgroundColor: liveRefresh ? '#4caf50' : '#6c757d',
                      color: '#fff',
                      transition: 'background-color 0.2s ease'
                    }}
                    title={liveRefresh ? 'Click to pause auto-refresh' : 'Click to enable auto-refresh'}
                  >
                    {liveRefresh ? '🔴 Live' : '⏸️ Auto'}
                  </button>

                  {/* Status Info */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    {liveRefresh && nextRefreshIn !== null && (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '12px',
                        color: '#2e7d32'
                      }}>
                        <span style={{
                          display: 'inline-block',
                          width: '8px',
                          height: '8px',
                          backgroundColor: '#4caf50',
                          borderRadius: '50%',
                          animation: 'pulse 1.5s infinite'
                        }}></span>
                        <span>Next: <strong>{formatCountdown(nextRefreshIn)}</strong></span>
                      </div>
                    )}
                    {!liveRefresh && parseBarSizeToMs(selectedBarSize) && (
                      <span style={{ fontSize: '11px', color: '#666' }}>
                        Auto-refresh: every {selectedBarSize}
                      </span>
                    )}
                    {lastRefreshTime && (
                      <span style={{ fontSize: '10px', color: '#999' }}>
                        Last: {new Date(lastRefreshTime).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {marketDataLoading ? (
              <div style={{ padding: '40px', textAlign: 'center', color: '#6c757d' }}>
                <div style={{ marginBottom: '10px' }}>Loading market data...</div>
                <div style={{ fontSize: '14px', color: '#999' }}>Aggregating and processing data...</div>
              </div>
            ) : marketData.length === 0 ? (
              <p>No market data available</p>
            ) : (
              <>
                <div style={{
                  marginBottom: '20px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{ color: '#6c757d' }}>
                    Showing {marketData.filter((md) => !selectedBarSize || md.bar_size === selectedBarSize).length} bars
                    {selectedBarSize && ` for ${selectedBarSize}`}
                  </div>
                </div>
                <MarketDataChart
                  data={marketData.filter((md) => !selectedBarSize || md.bar_size === selectedBarSize)}
                  selectedIndicators={selectedIndicators}
                />
                <div style={{ marginTop: '20px' }}>
                  <h4>Data Summary</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Bar Size</th>
                        <th>Bars Count</th>
                        <th>Latest Timestamp</th>
                        <th>Latest Close</th>
                        <th>Available Indicators</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Array.from(new Set(marketData.map((md) => md.bar_size))).map((barSize) => {
                        const barsForSize = marketData.filter((md) => md.bar_size === barSize)
                        const latest = barsForSize[0] // Already sorted by timestamp desc
                        const indicators = new Set()
                        barsForSize.forEach((md) => {
                          if (md.indicators) {
                            Object.keys(md.indicators).forEach((ind) => indicators.add(ind))
                          }
                        })
                        return (
                          <tr key={barSize}>
                            <td>{barSize}</td>
                            <td>{barsForSize.length}</td>
                            <td>{formatDate(latest.timestamp)}</td>
                            <td>{formatCurrency(latest.close)}</td>
                            <td>{Array.from(indicators).sort().join(', ') || 'None'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default OperationDetail

