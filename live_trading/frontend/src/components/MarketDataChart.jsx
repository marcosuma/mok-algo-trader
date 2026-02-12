/**
 * MarketDataChart Component
 *
 * Displays market data (OHLCV) with technical indicators in proper chart panels.
 *
 * Chart Layout:
 * - Main Price Chart: Candlestick/Line with price overlay indicators (MAs, Bollinger)
 * - Volume Chart: Bar chart showing trading volume
 * - Dynamic Indicator Charts: RSI, MACD, ADX, ATR each in separate panels
 *
 * Indicator Categories:
 * - PRICE_OVERLAY: Displayed on main price chart (same scale as price)
 *   - SMA_50, SMA_100, SMA_200, EMA_10, EMA_20, EMA_50
 *   - bollinger_up, bollinger_down, bollinger_mid
 * - OSCILLATOR_0_100: Separate chart with 0-100 scale
 *   - RSI_14, adx, plus_di, minus_di
 * - MACD: Separate chart with histogram
 *   - macd, macd_s, macd_h
 * - VOLATILITY: Separate chart for volatility measures
 *   - ATR_14, atr
 */
import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react'
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Area,
  Bar,
  Brush,
  ReferenceArea,
  ReferenceLine,
  Customized
} from 'recharts'
import { formatDate } from '../utils/formatters'

// Indicator configuration - defines how each indicator should be displayed
const INDICATOR_CONFIG = {
  // Price overlays - shown on main price chart
  SMA_50: { type: 'PRICE_OVERLAY', color: '#8884d8', name: 'SMA 50' },
  SMA_100: { type: 'PRICE_OVERLAY', color: '#82ca9d', name: 'SMA 100' },
  SMA_200: { type: 'PRICE_OVERLAY', color: '#ffc658', name: 'SMA 200' },
  EMA_10: { type: 'PRICE_OVERLAY', color: '#ff7300', name: 'EMA 10' },
  EMA_20: { type: 'PRICE_OVERLAY', color: '#00C49F', name: 'EMA 20' },
  EMA_50: { type: 'PRICE_OVERLAY', color: '#FFBB28', name: 'EMA 50' },
  bollinger_up: { type: 'PRICE_OVERLAY', color: '#ff6b6b', name: 'BB Upper', dashArray: '5 5' },
  bollinger_down: { type: 'PRICE_OVERLAY', color: '#4ecdc4', name: 'BB Lower', dashArray: '5 5' },
  bollinger_mid: { type: 'PRICE_OVERLAY', color: '#95a5a6', name: 'BB Mid', dashArray: '3 3' },

  // RSI - separate oscillator chart (0-100)
  RSI_14: { type: 'RSI', color: '#8884d8', name: 'RSI 14', chartTitle: 'RSI (14)' },

  // MACD - separate chart with signal and histogram
  macd: { type: 'MACD', color: '#2196F3', name: 'MACD' },
  macd_s: { type: 'MACD', color: '#FF9800', name: 'Signal' },
  macd_h: { type: 'MACD', color: '#4CAF50', name: 'Histogram', isHistogram: true },

  // ADX - separate oscillator chart (0-100)
  adx: { type: 'ADX', color: '#9C27B0', name: 'ADX', chartTitle: 'ADX' },
  plus_di: { type: 'ADX', color: '#4CAF50', name: '+DI' },
  minus_di: { type: 'ADX', color: '#F44336', name: '-DI' },

  // ATR - separate volatility chart
  ATR_14: { type: 'ATR', color: '#FF5722', name: 'ATR 14', chartTitle: 'ATR (14)' },
  atr: { type: 'ATR', color: '#FF5722', name: 'ATR', chartTitle: 'ATR' },
}

// Chart heights
const PRICE_CHART_HEIGHT = 400
const VOLUME_CHART_HEIGHT = 100
const INDICATOR_CHART_HEIGHT = 120

// Time range preset definitions
const TIME_RANGE_PRESETS = [
  { label: '1H', value: 60 * 60 * 1000, tooltip: 'Last 1 hour' },
  { label: '4H', value: 4 * 60 * 60 * 1000, tooltip: 'Last 4 hours' },
  { label: '24H', value: 24 * 60 * 60 * 1000, tooltip: 'Last 24 hours' },
  { label: '7D', value: 7 * 24 * 60 * 60 * 1000, tooltip: 'Last 7 days' },
  { label: '30D', value: 30 * 24 * 60 * 60 * 1000, tooltip: 'Last 30 days' },
  { label: '90D', value: 90 * 24 * 60 * 60 * 1000, tooltip: 'Last 90 days' },
  { label: 'All', value: null, tooltip: 'Show all data' },
]

// Helper to format a Date as a local datetime-local input value (YYYY-MM-DDTHH:mm)
const toDateTimeLocalString = (date) => {
  const d = new Date(date)
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function MarketDataChart({ data, selectedIndicators = [] }) {
  const [xAxisDomain, setXAxisDomain] = useState(['dataMin', 'dataMax'])
  const [chartType, setChartType] = useState('candlestick')
  const [isSelecting, setIsSelecting] = useState(false)
  const [selectionStart, setSelectionStart] = useState(null)
  const [selectionEnd, setSelectionEnd] = useState(null)
  const [hiddenSeries, setHiddenSeries] = useState(new Set())
  const [chartWidth, setChartWidth] = useState(800)
  const [activePreset, setActivePreset] = useState('All')
  const [customRangeOpen, setCustomRangeOpen] = useState(false)
  const [customStartDate, setCustomStartDate] = useState('')
  const [customEndDate, setCustomEndDate] = useState('')
  const containerRef = useRef(null)
  const previousDataRef = useRef(null)

  // Track chart container width
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        setChartWidth(containerRef.current.offsetWidth)
      }
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => window.removeEventListener('resize', updateWidth)
  }, [])

  // Handle time range preset selection
  const handlePresetSelect = useCallback((preset) => {
    setActivePreset(preset.label)
    setCustomRangeOpen(false)

    if (preset.value === null) {
      // "All" - show everything
      setXAxisDomain(['dataMin', 'dataMax'])
      return
    }

    // Calculate the end timestamp from the latest data point
    if (!data || data.length === 0) return
    const latestTimestamp = Math.max(...data.map(d => new Date(d.timestamp).getTime()))
    const startTimestamp = latestTimestamp - preset.value
    setXAxisDomain([startTimestamp, latestTimestamp])
  }, [data])

  // Handle custom date range application
  const handleCustomRangeApply = useCallback(() => {
    if (!customStartDate && !customEndDate) return

    const startTs = customStartDate ? new Date(customStartDate).getTime() : 'dataMin'
    const endTs = customEndDate ? new Date(customEndDate).getTime() : 'dataMax'

    setActivePreset('custom')
    setXAxisDomain([startTs, endTs])
  }, [customStartDate, customEndDate])

  // Initialize custom date range inputs from data bounds
  const dataTimeBounds = useMemo(() => {
    if (!data || data.length === 0) return null
    const timestamps = data.map(d => new Date(d.timestamp).getTime())
    return {
      min: Math.min(...timestamps),
      max: Math.max(...timestamps),
    }
  }, [data])

  // Dynamic max points based on chart width
  const getMaxPoints = (width) => {
    const pointsPerPixel = 0.5
    return Math.max(100, Math.floor(width * pointsPerPixel))
  }

  // Data downsampling function
  const downsampleData = (inputData, maxPoints) => {
    if (inputData.length <= maxPoints) return inputData

    const factor = Math.ceil(inputData.length / maxPoints)
    const downsampled = []

    for (let i = 0; i < inputData.length; i += factor) {
      const chunk = inputData.slice(i, Math.min(i + factor, inputData.length))
      if (chunk.length === 0) continue

      const last = chunk[chunk.length - 1]
      const aggregated = {
        open: chunk[0].open,
        high: Math.max(...chunk.map(d => d.high)),
        low: Math.min(...chunk.map(d => d.low)),
        close: last.close,
        volume: chunk.reduce((sum, d) => sum + (d.volume || 0), 0),
        timestamp: chunk[0].timestamp,
        time: chunk[0].time,
        index: Math.floor(i / factor),
        _aggregated: true,
        _aggregatedCount: chunk.length
      }

      // Copy all other fields from last point
      Object.keys(last).forEach(key => {
        if (!['open', 'high', 'low', 'close', 'volume', 'timestamp', 'time', 'index', '_aggregated', '_aggregatedCount'].includes(key)) {
          aggregated[key] = last[key]
        }
      })

      downsampled.push(aggregated)
    }

    return downsampled
  }

  // All data points without downsampling - deduplicated by timestamp
  const allChartData = useMemo(() => {
    if (!data || data.length === 0) return []

    // First pass: filter weekends and create points
    const rawPoints = data
      .slice()
      .reverse()
      .filter((item) => {
        const date = new Date(item.timestamp)
        const dayOfWeek = date.getDay()
        return dayOfWeek !== 0 && dayOfWeek !== 6
      })
      .map((item) => {
        const timestamp = new Date(item.timestamp).getTime()
        const point = {
          timestamp: timestamp,
          time: new Date(item.timestamp).toLocaleString(),
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume || 0,
        }

        // Add all indicators
        if (item.indicators) {
          Object.keys(item.indicators).forEach((indicator) => {
            if (item.indicators[indicator] !== undefined) {
              point[indicator] = item.indicators[indicator]
            }
          })
        }

        return point
      })

    // Deduplicate by timestamp - keep last occurrence for each timestamp
    const timestampMap = new Map()
    rawPoints.forEach((point) => {
      timestampMap.set(point.timestamp, point)
    })

    // Convert back to array and add sequential indices
    return Array.from(timestampMap.values())
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((point, index) => ({
        ...point,
        index: index,
        _uniqueKey: `${point.timestamp}-${index}` // Unique key for React
      }))
  }, [data])

  // Visible chart data with downsampling
  const chartData = useMemo(() => {
    if (allChartData.length === 0) return []

    let visibleData = allChartData

    if (xAxisDomain[0] !== 'dataMin' || xAxisDomain[1] !== 'dataMax') {
      const minTs = xAxisDomain[0] === 'dataMin' ? -Infinity : xAxisDomain[0]
      const maxTs = xAxisDomain[1] === 'dataMax' ? Infinity : xAxisDomain[1]
      visibleData = allChartData.filter(d => d.timestamp >= minTs && d.timestamp <= maxTs)
    }

    // If no data after filtering, return empty
    if (visibleData.length === 0) return []

    const maxPoints = getMaxPoints(chartWidth)

    if (visibleData.length > maxPoints) {
      const downsampled = downsampleData(visibleData, maxPoints)
      return downsampled.map((point, idx) => ({
        ...point,
        index: idx,
        _uniqueKey: `agg-${point.timestamp}-${idx}`
      }))
    }

    // Assign sequential indices for the visible range
    return visibleData.map((point, idx) => ({
      ...point,
      index: idx,
      _uniqueKey: `${point.timestamp}-${idx}`
    }))
  }, [allChartData, xAxisDomain, chartWidth])

  // Reset zoom on data change
  useEffect(() => {
    const isInitialLoad = previousDataRef.current === null
    const wasEmpty = previousDataRef.current && previousDataRef.current.length === 0
    const isNowPopulated = data.length > 0

    if (isInitialLoad || (wasEmpty && isNowPopulated)) {
      setXAxisDomain(['dataMin', 'dataMax'])
      setIsSelecting(false)
      setSelectionStart(null)
      setSelectionEnd(null)
    }

    previousDataRef.current = data
  }, [data])

  // Categorize selected indicators
  const categorizedIndicators = useMemo(() => {
    const result = {
      priceOverlay: [],
      rsi: [],
      macd: [],
      adx: [],
      atr: [],
    }

    selectedIndicators.forEach(indicator => {
      const config = INDICATOR_CONFIG[indicator]
      if (!config) return

      switch (config.type) {
        case 'PRICE_OVERLAY':
          result.priceOverlay.push({ key: indicator, ...config })
          break
        case 'RSI':
          result.rsi.push({ key: indicator, ...config })
          break
        case 'MACD':
          result.macd.push({ key: indicator, ...config })
          break
        case 'ADX':
          result.adx.push({ key: indicator, ...config })
          break
        case 'ATR':
          result.atr.push({ key: indicator, ...config })
          break
        default:
          result.priceOverlay.push({ key: indicator, ...config })
      }
    })

    return result
  }, [selectedIndicators])

  // X-axis domain calculations - simplified to always match chartData indices
  const xAxisIndexDomain = useMemo(() => {
    if (chartData.length === 0) return [0, 0]
    // Since chartData is already filtered and re-indexed from 0,
    // the domain should always be [0, length-1]
    return [0, chartData.length - 1]
  }, [chartData])

  // Brush indices - tracks the current selection in the full data range
  const brushIndices = useMemo(() => {
    if (allChartData.length === 0) return { startIndex: 0, endIndex: 0 }

    // If showing all data, select entire range
    if (xAxisDomain[0] === 'dataMin' && xAxisDomain[1] === 'dataMax') {
      return { startIndex: 0, endIndex: allChartData.length - 1 }
    }

    // Find indices in allChartData that match the current domain
    const minTs = xAxisDomain[0]
    const maxTs = xAxisDomain[1]

    let startIdx = allChartData.findIndex(d => d.timestamp >= minTs)
    let endIdx = allChartData.length - 1

    // Find last index within range
    for (let i = allChartData.length - 1; i >= 0; i--) {
      if (allChartData[i].timestamp <= maxTs) {
        endIdx = i
        break
      }
    }

    if (startIdx === -1) startIdx = 0

    return { startIndex: startIdx, endIndex: endIdx }
  }, [allChartData, xAxisDomain])

  // Event handlers
  const handleResetZoom = () => {
    setXAxisDomain(['dataMin', 'dataMax'])
    setActivePreset('All')
    setIsSelecting(false)
    setSelectionStart(null)
    setSelectionEnd(null)
  }

  const handleDoubleClick = () => handleResetZoom()

  const handleChartMouseDown = (e) => {
    if (!e || chartData.length === 0) return
    let timestamp = null
    if (e.activePayload && e.activePayload.length > 0) {
      const dataPoint = e.activePayload[0].payload
      if (dataPoint && dataPoint.timestamp) timestamp = dataPoint.timestamp
    }
    if (timestamp !== null && !isNaN(timestamp)) {
      setIsSelecting(true)
      setSelectionStart(timestamp)
      setSelectionEnd(timestamp)
    }
  }

  const handleChartMouseMove = (e) => {
    if (!isSelecting || selectionStart === null || !e || chartData.length === 0) return
    let timestamp = null
    if (e.activePayload && e.activePayload.length > 0) {
      const dataPoint = e.activePayload[0].payload
      if (dataPoint && dataPoint.timestamp) timestamp = dataPoint.timestamp
    }
    if (timestamp !== null && !isNaN(timestamp)) setSelectionEnd(timestamp)
  }

  const handleChartMouseUp = () => {
    if (!isSelecting || selectionStart === null || selectionEnd === null) {
      setIsSelecting(false)
      setSelectionStart(null)
      setSelectionEnd(null)
      return
    }

    const start = Math.min(selectionStart, selectionEnd)
    const end = Math.max(selectionStart, selectionEnd)
    // Get current range from chartData (already indexed 0 to length-1)
    const currentRange = chartData.length > 0 ? chartData.length - 1 : 0
    const startPoint = chartData.find((d) => d.timestamp >= start)
    const endPoint = [...chartData].reverse().find((d) => d.timestamp <= end)

    if (startPoint && endPoint) {
      const selectionRange = Math.abs(endPoint.index - startPoint.index)
      // Only zoom if selection is meaningful (at least 1% of current range or at least 2 points)
      if (selectionRange > Math.max(currentRange * 0.01, 1)) {
        setXAxisDomain([start, end])
        setActivePreset('custom')
      }
    }

    setIsSelecting(false)
    setSelectionStart(null)
    setSelectionEnd(null)
  }

  const handleChartMouseLeave = () => {
    if (isSelecting) {
      setIsSelecting(false)
      setSelectionStart(null)
      setSelectionEnd(null)
    }
  }

  const handleBrushChange = useCallback((domain) => {
    if (!domain || allChartData.length === 0) return

    let startArrayIndex, endArrayIndex
    if (domain.startIndex !== undefined && domain.endIndex !== undefined) {
      startArrayIndex = Math.floor(domain.startIndex)
      endArrayIndex = Math.ceil(domain.endIndex)
    } else if (domain.startValue !== undefined && domain.endValue !== undefined) {
      startArrayIndex = Math.round(domain.startValue)
      endArrayIndex = Math.round(domain.endValue)
    } else return

    // Use allChartData for index lookup since brush uses full data
    startArrayIndex = Math.max(0, Math.min(startArrayIndex, allChartData.length - 1))
    endArrayIndex = Math.max(0, Math.min(endArrayIndex, allChartData.length - 1))

    const startPoint = allChartData[startArrayIndex]
    const endPoint = allChartData[endArrayIndex]

    if (startPoint && endPoint) {
      setXAxisDomain([startPoint.timestamp, endPoint.timestamp])
      setActivePreset('custom')
    }
  }, [allChartData])

  if (chartData.length === 0) {
    return <div style={{ padding: '40px', textAlign: 'center', color: '#6c757d' }}>No market data available</div>
  }

  // Shared X-axis tick formatter
  const xAxisTickFormatter = (value) => {
    const dataPoint = chartData.find((d) => d.index === Math.round(value))
    if (dataPoint) {
      const date = new Date(dataPoint.timestamp)
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    }
    return ''
  }

  // Common chart props for synchronized charts
  const commonChartProps = {
    data: chartData,
    margin: { top: 5, right: 50, left: 50, bottom: 5 },
    onMouseDown: handleChartMouseDown,
    onMouseMove: handleChartMouseMove,
    onMouseUp: handleChartMouseUp,
    onMouseLeave: handleChartMouseLeave,
  }

  const commonXAxisProps = {
    dataKey: 'index',
    type: 'number',
    domain: xAxisIndexDomain,
    tickFormatter: xAxisTickFormatter,
    tick: { fill: chartType === 'candlestick' ? '#888' : '#666', fontSize: 10 },
    axisLine: { stroke: chartType === 'candlestick' ? '#444' : '#ccc' },
    tickLine: { stroke: chartType === 'candlestick' ? '#444' : '#ccc' },
    hide: true, // Hide X axis on all but last chart
  }

  const isDarkTheme = chartType === 'candlestick'
  const bgColor = isDarkTheme ? '#1a1a2e' : '#fafafa'
  const gridColor = isDarkTheme ? '#2a2a3e' : '#e0e0e0'
  const textColor = isDarkTheme ? '#888' : '#666'

  // Custom tooltip for all charts with OHLC data and price change
  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null

    const dataPoint = payload[0]?.payload
    if (!dataPoint) return null

    const dateString = dataPoint.time || new Date(dataPoint.timestamp).toLocaleString()

    // Find previous data point for calculating change
    const currentIndex = dataPoint.index
    const previousDataPoint = chartData.find(d => d.index === currentIndex - 1)

    // Calculate price change
    let priceChange = null
    let priceChangePercent = null
    let changeColor = '#888'

    if (previousDataPoint) {
      priceChange = dataPoint.close - previousDataPoint.close
      priceChangePercent = ((priceChange / previousDataPoint.close) * 100)
      changeColor = priceChange >= 0 ? '#26a69a' : '#ef5350'
    }

    // Determine if this tooltip is for the price chart (has OHLC data in payload or datapoint)
    const isPriceChart = dataPoint.open !== undefined && dataPoint.high !== undefined

    // Separate indicators from base data
    const indicatorEntries = payload.filter(entry =>
      !['close', 'open', 'high', 'low', 'volume', 'Price'].includes(entry.name) &&
      entry.name !== 'Volume'
    )

    return (
      <div style={{
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        border: '1px solid #444',
        borderRadius: '8px',
        padding: '12px 16px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
        color: '#fff',
        fontSize: '12px',
        minWidth: '180px',
        backdropFilter: 'blur(8px)'
      }}>
        {/* Date Header */}
        <div style={{
          marginBottom: '10px',
          paddingBottom: '8px',
          borderBottom: '1px solid #333',
          fontWeight: '600',
          color: '#ccc'
        }}>
          📅 {dateString}
        </div>

        {/* OHLC Data */}
        {isPriceChart && (
          <div style={{ marginBottom: '10px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#888' }}>Open:</span>
                <span style={{ color: '#fff', fontWeight: '500' }}>{dataPoint.open?.toFixed(5)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#888' }}>High:</span>
                <span style={{ color: '#26a69a', fontWeight: '500' }}>{dataPoint.high?.toFixed(5)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#888' }}>Low:</span>
                <span style={{ color: '#ef5350', fontWeight: '500' }}>{dataPoint.low?.toFixed(5)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#888' }}>Close:</span>
                <span style={{ color: '#fff', fontWeight: '500' }}>{dataPoint.close?.toFixed(5)}</span>
              </div>
            </div>
          </div>
        )}

        {/* Price Change from Previous */}
        {isPriceChart && priceChange !== null && (
          <div style={{
            marginBottom: '10px',
            padding: '8px',
            backgroundColor: 'rgba(255,255,255,0.05)',
            borderRadius: '4px',
            borderLeft: `3px solid ${changeColor}`
          }}>
            <div style={{ color: '#888', fontSize: '10px', marginBottom: '4px', textTransform: 'uppercase' }}>
              Change from Previous
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ color: changeColor, fontWeight: '600', fontSize: '14px' }}>
                {priceChange >= 0 ? '▲' : '▼'} {Math.abs(priceChange).toFixed(5)}
              </span>
              <span style={{
                color: changeColor,
                fontWeight: '600',
                backgroundColor: priceChange >= 0 ? 'rgba(38,166,154,0.2)' : 'rgba(239,83,80,0.2)',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '11px'
              }}>
                {priceChange >= 0 ? '+' : ''}{priceChangePercent.toFixed(3)}%
              </span>
            </div>
          </div>
        )}

        {/* Volume */}
        {isPriceChart && dataPoint.volume !== undefined && (
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginBottom: indicatorEntries.length > 0 ? '10px' : '0',
            paddingBottom: indicatorEntries.length > 0 ? '8px' : '0',
            borderBottom: indicatorEntries.length > 0 ? '1px solid #333' : 'none'
          }}>
            <span style={{ color: '#888' }}>📊 Volume:</span>
            <span style={{ color: '#aaa', fontWeight: '500' }}>
              {dataPoint.volume >= 1000000
                ? `${(dataPoint.volume/1000000).toFixed(2)}M`
                : dataPoint.volume >= 1000
                  ? `${(dataPoint.volume/1000).toFixed(1)}K`
                  : dataPoint.volume?.toLocaleString()}
            </span>
          </div>
        )}

        {/* Indicators */}
        {indicatorEntries.length > 0 && (
          <div>
            <div style={{ color: '#888', fontSize: '10px', marginBottom: '6px', textTransform: 'uppercase' }}>
              Indicators
            </div>
            {indicatorEntries.map((entry, index) => (
              <div key={index} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                margin: '3px 0'
              }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{
                    width: '8px',
                    height: '8px',
                    backgroundColor: entry.color,
                    borderRadius: '2px',
                    display: 'inline-block'
                  }}></span>
                  <span style={{ color: '#aaa' }}>{entry.name}:</span>
                </span>
                <span style={{ color: entry.color, fontWeight: '500' }}>
                  {typeof entry.value === 'number' ? entry.value.toFixed(entry.name === 'Volume' ? 0 : 5) : entry.value}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Non-price chart data (for indicator panels) */}
        {!isPriceChart && indicatorEntries.length === 0 && payload.map((entry, index) => (
          <div key={index} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            margin: '3px 0'
          }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{
                width: '8px',
                height: '8px',
                backgroundColor: entry.color,
                borderRadius: '2px',
                display: 'inline-block'
              }}></span>
              <span style={{ color: '#aaa' }}>{entry.name}:</span>
            </span>
            <span style={{ color: entry.color, fontWeight: '500' }}>
              {typeof entry.value === 'number' ? entry.value.toFixed(entry.name === 'Volume' ? 0 : 5) : entry.value}
            </span>
          </div>
        ))}

        {/* Aggregation indicator */}
        {dataPoint._aggregated && (
          <div style={{
            marginTop: '8px',
            paddingTop: '8px',
            borderTop: '1px solid #333',
            fontSize: '10px',
            color: '#f0ad4e',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}>
            ⚡ Aggregated from {dataPoint._aggregatedCount} bars
          </div>
        )}
      </div>
    )
  }

  // Render candlesticks
  const renderCandlesticks = ({ xAxisMap, yAxisMap, offset }) => {
    const xAxis = xAxisMap && (xAxisMap[0] || Object.values(xAxisMap)[0])
    const yAxis = yAxisMap && yAxisMap.price

    if (!xAxis?.scale || !yAxis?.scale) return null
    if (chartData.length === 0) return null

    const xScale = xAxis.scale
    const yScale = yAxis.scale
    const currentChartWidth = offset?.width || 700
    const numCandles = chartData.length
    const availableWidth = currentChartWidth * 0.9
    const rawCandleWidth = availableWidth / Math.max(numCandles, 1)
    const candleWidth = Math.max(4, Math.min(30, rawCandleWidth * 0.7))
    const wickWidth = Math.max(1, Math.min(3, candleWidth * 0.15))

    // Calculate candle positions directly to avoid scale issues
    const chartLeft = offset?.left || 50
    const chartRight = (offset?.left || 50) + (offset?.width || 700)
    const usableWidth = chartRight - chartLeft

    return (
      <g className="candlesticks">
        {chartData.map((d, i) => {
          // Use direct position calculation as backup if scale gives bad results
          let x = xScale(d.index)

          // Validate x position - if invalid, calculate directly
          if (typeof x !== 'number' || isNaN(x) || x < chartLeft - 50 || x > chartRight + 50) {
            // Direct calculation: distribute candles evenly across the chart width
            if (numCandles === 1) {
              x = chartLeft + usableWidth / 2
            } else {
              x = chartLeft + (d.index / (numCandles - 1)) * usableWidth
            }
          }

          const yHigh = yScale(d.high)
          const yLow = yScale(d.low)
          const yOpen = yScale(d.open)
          const yClose = yScale(d.close)

          if (typeof yHigh !== 'number' || isNaN(yHigh)) return null

          const isUp = d.close >= d.open
          const color = isUp ? '#26a69a' : '#ef5350'
          const bodyTop = Math.min(yOpen, yClose)
          const bodyBottom = Math.max(yOpen, yClose)
          const bodyHeight = Math.max(bodyBottom - bodyTop, 2)

          // Use unique key based on timestamp to ensure proper React reconciliation
          const uniqueKey = d._uniqueKey || `candle-${d.timestamp}-${i}`

          return (
            <g key={uniqueKey}>
              <line x1={x} y1={yHigh} x2={x} y2={bodyTop} stroke={color} strokeWidth={wickWidth} />
              <line x1={x} y1={bodyBottom} x2={x} y2={yLow} stroke={color} strokeWidth={wickWidth} />
              <rect
                x={x - candleWidth / 2}
                y={bodyTop}
                width={candleWidth}
                height={bodyHeight}
                fill={color}
                stroke={color}
                strokeWidth={1}
                rx={1}
                ry={1}
              />
            </g>
          )
        })}
      </g>
    )
  }

  return (
    <div style={{ width: '100%', marginTop: '20px' }} ref={containerRef}>
      {/* Chart Toolbar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '10px',
        gap: '12px',
        flexWrap: 'wrap',
      }}>
        {/* Time Range Presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ fontSize: '12px', color: '#888', fontWeight: '600', marginRight: '4px' }}>Range:</span>
          {TIME_RANGE_PRESETS.map((preset) => (
            <button
              key={preset.label}
              onClick={() => handlePresetSelect(preset)}
              title={preset.tooltip}
              style={{
                padding: '5px 10px',
                border: activePreset === preset.label ? '1px solid #6366f1' : '1px solid #ddd',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: activePreset === preset.label ? '700' : '500',
                fontSize: '12px',
                backgroundColor: activePreset === preset.label ? '#6366f1' : 'transparent',
                color: activePreset === preset.label ? '#fff' : '#555',
                transition: 'all 0.15s ease',
                minWidth: '36px',
                textAlign: 'center',
              }}
            >
              {preset.label}
            </button>
          ))}

          {/* Custom Range Toggle */}
          <button
            onClick={() => setCustomRangeOpen(!customRangeOpen)}
            title="Custom date range"
            style={{
              padding: '5px 10px',
              border: activePreset === 'custom' || customRangeOpen ? '1px solid #6366f1' : '1px solid #ddd',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: activePreset === 'custom' || customRangeOpen ? '700' : '500',
              fontSize: '12px',
              backgroundColor: activePreset === 'custom' || customRangeOpen ? '#6366f1' : 'transparent',
              color: activePreset === 'custom' || customRangeOpen ? '#fff' : '#555',
              transition: 'all 0.15s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            Custom
          </button>
        </div>

        {/* Chart Type Toggle */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            onClick={() => setChartType('candlestick')}
            style={{
              padding: '5px 12px',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: chartType === 'candlestick' ? '600' : '400',
              fontSize: '12px',
              backgroundColor: chartType === 'candlestick' ? '#1a1a2e' : '#f0f0f0',
              color: chartType === 'candlestick' ? '#fff' : '#333',
              transition: 'all 0.2s ease',
            }}
          >
            Candlestick
          </button>
          <button
            onClick={() => setChartType('line')}
            style={{
              padding: '5px 12px',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: chartType === 'line' ? '600' : '400',
              fontSize: '12px',
              backgroundColor: chartType === 'line' ? '#1a1a2e' : '#f0f0f0',
              color: chartType === 'line' ? '#fff' : '#333',
              transition: 'all 0.2s ease',
            }}
          >
            Line
          </button>
        </div>
      </div>

      {/* Custom Date Range Picker */}
      {customRangeOpen && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          marginBottom: '10px',
          padding: '10px 14px',
          backgroundColor: '#f8f9fa',
          borderRadius: '6px',
          border: '1px solid #e0e0e0',
          flexWrap: 'wrap',
        }}>
          <label style={{ fontSize: '12px', fontWeight: '600', color: '#555' }}>From:</label>
          <input
            type="datetime-local"
            value={customStartDate}
            onChange={(e) => setCustomStartDate(e.target.value)}
            min={dataTimeBounds ? toDateTimeLocalString(dataTimeBounds.min) : undefined}
            max={dataTimeBounds ? toDateTimeLocalString(dataTimeBounds.max) : undefined}
            style={{
              padding: '5px 8px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '12px',
              color: '#333',
            }}
          />
          <label style={{ fontSize: '12px', fontWeight: '600', color: '#555' }}>To:</label>
          <input
            type="datetime-local"
            value={customEndDate}
            onChange={(e) => setCustomEndDate(e.target.value)}
            min={dataTimeBounds ? toDateTimeLocalString(dataTimeBounds.min) : undefined}
            max={dataTimeBounds ? toDateTimeLocalString(dataTimeBounds.max) : undefined}
            style={{
              padding: '5px 8px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              fontSize: '12px',
              color: '#333',
            }}
          />
          <button
            onClick={handleCustomRangeApply}
            disabled={!customStartDate && !customEndDate}
            style={{
              padding: '5px 14px',
              border: 'none',
              borderRadius: '4px',
              cursor: (!customStartDate && !customEndDate) ? 'not-allowed' : 'pointer',
              fontWeight: '600',
              fontSize: '12px',
              backgroundColor: (!customStartDate && !customEndDate) ? '#ccc' : '#6366f1',
              color: '#fff',
              transition: 'all 0.15s ease',
            }}
          >
            Apply
          </button>
          <button
            onClick={() => {
              setCustomStartDate('')
              setCustomEndDate('')
              setCustomRangeOpen(false)
              handleResetZoom()
            }}
            style={{
              padding: '5px 10px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: '500',
              fontSize: '12px',
              backgroundColor: 'transparent',
              color: '#666',
              transition: 'all 0.15s ease',
            }}
          >
            Clear
          </button>
          {dataTimeBounds && (
            <span style={{ fontSize: '11px', color: '#999', marginLeft: '4px' }}>
              Data: {new Date(dataTimeBounds.min).toLocaleDateString()} - {new Date(dataTimeBounds.max).toLocaleDateString()}
            </span>
          )}
        </div>
      )}

      {/* Main Container */}
      <div
        style={{
          backgroundColor: bgColor,
          borderRadius: '8px',
          border: isDarkTheme ? '1px solid #2a2a3e' : '1px solid #e0e0e0',
          overflow: 'visible',
        }}
        onDoubleClick={handleDoubleClick}
      >
        {/* Price Chart */}
        <div style={{ height: PRICE_CHART_HEIGHT, cursor: isSelecting ? 'crosshair' : 'default' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart {...commonChartProps}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis {...commonXAxisProps} />
              <YAxis
                yAxisId="price"
                orientation="right"
                domain={['auto', 'auto']}
                tick={{ fill: textColor, fontSize: 10 }}
                axisLine={{ stroke: gridColor }}
                tickLine={{ stroke: gridColor }}
                tickFormatter={(v) => v.toFixed(4)}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: '10px', color: textColor }} />

              {/* Candlestick or Line */}
              {chartType === 'candlestick' ? (
                <>
                  <Customized component={renderCandlesticks} />
                  <Line yAxisId="price" type="monotone" dataKey="close" stroke="transparent" strokeWidth={15} dot={false} name="Price" isAnimationActive={false} activeDot={{ r: 5, fill: '#fff', stroke: '#26a69a', strokeWidth: 2 }} />
                </>
              ) : (
                <>
                  <Line yAxisId="price" type="monotone" dataKey="close" stroke="#333" strokeWidth={2} dot={false} name="Close" isAnimationActive={false} />
                  <Line yAxisId="price" type="monotone" dataKey="open" stroke="#999" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Open" isAnimationActive={false} />
                  <Line yAxisId="price" type="monotone" dataKey="high" stroke="#26a69a" strokeWidth={1} dot={false} name="High" isAnimationActive={false} />
                  <Line yAxisId="price" type="monotone" dataKey="low" stroke="#ef5350" strokeWidth={1} dot={false} name="Low" isAnimationActive={false} />
                </>
              )}

              {/* Price Overlay Indicators */}
              {categorizedIndicators.priceOverlay.map((ind) => (
                <Line
                  key={ind.key}
                  yAxisId="price"
                  type="monotone"
                  dataKey={ind.key}
                  stroke={ind.color}
                  strokeWidth={1.5}
                  strokeDasharray={ind.dashArray}
                  dot={false}
                  name={ind.name}
                  isAnimationActive={false}
                />
              ))}

              {/* Selection area */}
              {isSelecting && selectionStart !== null && selectionEnd !== null && (() => {
                const startPoint = chartData.find((d) => d.timestamp >= Math.min(selectionStart, selectionEnd))
                const endPoint = [...chartData].reverse().find((d) => d.timestamp <= Math.max(selectionStart, selectionEnd))
                if (startPoint && endPoint) {
                  return <ReferenceArea yAxisId="price" x1={startPoint.index} x2={endPoint.index} strokeOpacity={0.3} fill="#8884d8" fillOpacity={0.1} />
                }
                return null
              })()}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Volume Chart */}
        <div style={{ height: VOLUME_CHART_HEIGHT, borderTop: `1px solid ${gridColor}` }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart {...commonChartProps} margin={{ ...commonChartProps.margin, top: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
              <XAxis {...commonXAxisProps} />
              <YAxis
                yAxisId="volume"
                orientation="right"
                domain={[0, 'auto']}
                tick={{ fill: textColor, fontSize: 9 }}
                axisLine={{ stroke: gridColor }}
                tickLine={{ stroke: gridColor }}
                tickFormatter={(v) => v >= 1000000 ? `${(v/1000000).toFixed(1)}M` : v >= 1000 ? `${(v/1000).toFixed(0)}K` : v}
                width={45}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar
                yAxisId="volume"
                dataKey="volume"
                name="Volume"
                isAnimationActive={false}
                shape={(props) => {
                  const { x, y, width, height, payload } = props
                  const isUp = payload.close >= payload.open
                  return <rect x={x} y={y} width={width} height={height} fill={isUp ? 'rgba(38, 166, 154, 0.6)' : 'rgba(239, 83, 80, 0.6)'} />
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
          <div style={{ textAlign: 'center', fontSize: '10px', color: textColor, marginTop: '-5px' }}>Volume</div>
        </div>

        {/* RSI Chart */}
        {categorizedIndicators.rsi.length > 0 && (
          <div style={{ height: INDICATOR_CHART_HEIGHT, borderTop: `1px solid ${gridColor}` }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart {...commonChartProps} margin={{ ...commonChartProps.margin, top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis {...commonXAxisProps} />
                <YAxis
                  yAxisId="rsi"
                  orientation="right"
                  domain={[0, 100]}
                  ticks={[30, 50, 70]}
                  tick={{ fill: textColor, fontSize: 9 }}
                  axisLine={{ stroke: gridColor }}
                  tickLine={{ stroke: gridColor }}
                  width={45}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine yAxisId="rsi" y={70} stroke="#ef5350" strokeDasharray="3 3" strokeOpacity={0.5} />
                <ReferenceLine yAxisId="rsi" y={30} stroke="#26a69a" strokeDasharray="3 3" strokeOpacity={0.5} />
                <ReferenceLine yAxisId="rsi" y={50} stroke={gridColor} strokeDasharray="2 2" />
                {categorizedIndicators.rsi.map((ind) => (
                  <Line key={ind.key} yAxisId="rsi" type="monotone" dataKey={ind.key} stroke={ind.color} strokeWidth={1.5} dot={false} name={ind.name} isAnimationActive={false} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: '10px', color: textColor, marginTop: '-5px' }}>RSI (14)</div>
          </div>
        )}

        {/* MACD Chart */}
        {categorizedIndicators.macd.length > 0 && (
          <div style={{ height: INDICATOR_CHART_HEIGHT, borderTop: `1px solid ${gridColor}` }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart {...commonChartProps} margin={{ ...commonChartProps.margin, top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis {...commonXAxisProps} />
                <YAxis
                  yAxisId="macd"
                  orientation="right"
                  domain={['auto', 'auto']}
                  tick={{ fill: textColor, fontSize: 9 }}
                  axisLine={{ stroke: gridColor }}
                  tickLine={{ stroke: gridColor }}
                  tickFormatter={(v) => v.toFixed(4)}
                  width={55}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine yAxisId="macd" y={0} stroke={gridColor} />
                {/* MACD Histogram as bars */}
                {categorizedIndicators.macd.filter(ind => ind.isHistogram).map((ind) => (
                  <Bar
                    key={ind.key}
                    yAxisId="macd"
                    dataKey={ind.key}
                    name={ind.name}
                    isAnimationActive={false}
                    shape={(props) => {
                      const { x, y, width, height, payload } = props
                      const value = payload[ind.key]
                      const isPositive = value >= 0
                      return <rect x={x} y={y} width={width} height={Math.abs(height)} fill={isPositive ? 'rgba(38, 166, 154, 0.7)' : 'rgba(239, 83, 80, 0.7)'} />
                    }}
                  />
                ))}
                {/* MACD and Signal lines */}
                {categorizedIndicators.macd.filter(ind => !ind.isHistogram).map((ind) => (
                  <Line key={ind.key} yAxisId="macd" type="monotone" dataKey={ind.key} stroke={ind.color} strokeWidth={1.5} dot={false} name={ind.name} isAnimationActive={false} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: '10px', color: textColor, marginTop: '-5px' }}>MACD</div>
          </div>
        )}

        {/* ADX Chart */}
        {categorizedIndicators.adx.length > 0 && (
          <div style={{ height: INDICATOR_CHART_HEIGHT, borderTop: `1px solid ${gridColor}` }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart {...commonChartProps} margin={{ ...commonChartProps.margin, top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis {...commonXAxisProps} />
                <YAxis
                  yAxisId="adx"
                  orientation="right"
                  domain={[0, 100]}
                  ticks={[25, 50, 75]}
                  tick={{ fill: textColor, fontSize: 9 }}
                  axisLine={{ stroke: gridColor }}
                  tickLine={{ stroke: gridColor }}
                  width={45}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine yAxisId="adx" y={25} stroke={textColor} strokeDasharray="3 3" strokeOpacity={0.5} />
                {categorizedIndicators.adx.map((ind) => (
                  <Line key={ind.key} yAxisId="adx" type="monotone" dataKey={ind.key} stroke={ind.color} strokeWidth={1.5} dot={false} name={ind.name} isAnimationActive={false} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: '10px', color: textColor, marginTop: '-5px' }}>ADX</div>
          </div>
        )}

        {/* ATR Chart */}
        {categorizedIndicators.atr.length > 0 && (
          <div style={{ height: INDICATOR_CHART_HEIGHT, borderTop: `1px solid ${gridColor}` }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart {...commonChartProps} margin={{ ...commonChartProps.margin, top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                <XAxis {...commonXAxisProps} />
                <YAxis
                  yAxisId="atr"
                  orientation="right"
                  domain={[0, 'auto']}
                  tick={{ fill: textColor, fontSize: 9 }}
                  axisLine={{ stroke: gridColor }}
                  tickLine={{ stroke: gridColor }}
                  tickFormatter={(v) => v.toFixed(5)}
                  width={55}
                />
                <Tooltip content={<CustomTooltip />} />
                {categorizedIndicators.atr.map((ind) => (
                  <Area key={ind.key} yAxisId="atr" type="monotone" dataKey={ind.key} stroke={ind.color} fill={ind.color} fillOpacity={0.2} strokeWidth={1.5} dot={false} name={ind.name} isAnimationActive={false} />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: '10px', color: textColor, marginTop: '-5px' }}>ATR</div>
          </div>
        )}

        {/* Brush (Time Range Selector) - Uses allChartData for full range */}
        <div
          style={{
            height: 60,
            padding: '10px 50px',
            borderTop: `1px solid ${gridColor}`,
            backgroundColor: isDarkTheme ? '#16162a' : '#f0f0f0',
          }}
          onDoubleClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={allChartData}
              margin={{ top: 0, right: 0, left: 0, bottom: 0 }}
            >
              <Brush
                dataKey="index"
                height={40}
                stroke={isDarkTheme ? '#6366f1' : '#8884d8'}
                fill={isDarkTheme ? '#1e1e38' : '#fff'}
                startIndex={brushIndices.startIndex}
                endIndex={brushIndices.endIndex}
                onChange={handleBrushChange}
                travellerWidth={12}
                gap={1}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Info Footer */}
      <div style={{ marginTop: '10px', fontSize: '12px', color: '#6c757d', textAlign: 'center' }}>
        Drag to zoom | Double-click to reset | Use range presets above
        {chartData.length > 0 && (
          <span style={{ display: 'block', marginTop: '4px' }}>
            {chartData.some(d => d._aggregated) ? (
              <span style={{ color: '#f0ad4e' }}>Showing {chartData.length} aggregated bars (zoom in for detail)</span>
            ) : (
              <span style={{ color: '#5cb85c' }}>Showing {chartData.length} of {allChartData.length} data points</span>
            )}
            {activePreset !== 'All' && activePreset !== 'custom' && (
              <span style={{ color: '#6366f1', marginLeft: '8px' }}>
                [{activePreset} range]
              </span>
            )}
            {activePreset === 'custom' && (
              <span style={{ color: '#6366f1', marginLeft: '8px' }}>
                [custom range]
              </span>
            )}
          </span>
        )}
      </div>
    </div>
  )
}

export default MarketDataChart
