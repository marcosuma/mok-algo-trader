import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import StrategyConfigForm, { toLabel, inferInputType, inferStep } from './StrategyConfigForm'
import { strategiesApi } from '../api/client'

vi.mock('../api/client', () => ({
  strategiesApi: {
    list: vi.fn(),
  },
}))

const ATR_BREAKOUT_DEFAULTS = {
  lookback_period: 20,
  atr_multiplier: 1.5,
}

const IFS_DEFAULTS = {
  hurst_threshold: 0.52,
  use_rsi_filter: true,
  adx_min: 25.0,
  cooldown_bars: 8,
}

beforeEach(() => {
  vi.clearAllMocks()
})

// ---------------------------------------------------------------------------
// Pure helper tests
// ---------------------------------------------------------------------------

describe('toLabel', () => {
  it('converts snake_case to Title Case', () => {
    expect(toLabel('lookback_period')).toBe('Lookback Period')
    expect(toLabel('atr_multiplier')).toBe('Atr Multiplier')
    expect(toLabel('use_rsi_filter')).toBe('Use Rsi Filter')
  })
})

describe('inferInputType', () => {
  it('returns checkbox for booleans', () => expect(inferInputType(true)).toBe('checkbox'))
  it('returns number for numbers', () => expect(inferInputType(1.5)).toBe('number'))
  it('returns text for strings', () => expect(inferInputType('foo')).toBe('text'))
})

describe('inferStep', () => {
  it('returns 1 for integers', () => expect(inferStep(20)).toBe('1'))
  it('returns 0.01 for floats', () => expect(inferStep(1.5)).toBe('0.01'))
})

// ---------------------------------------------------------------------------
// Component rendering
// ---------------------------------------------------------------------------

describe('StrategyConfigForm', () => {
  it('renders nothing when strategyName is empty', () => {
    const { container } = render(
      <StrategyConfigForm strategyName="" config={{}} onChange={() => {}} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders numeric fields for each non-boolean param', async () => {
    strategiesApi.list.mockResolvedValue({
      data: [{ name: 'ATRBreakout', default_config: ATR_BREAKOUT_DEFAULTS }],
    })
    const onChange = vi.fn()

    render(<StrategyConfigForm strategyName="ATRBreakout" config={{}} onChange={onChange} />)

    await waitFor(() => {
      expect(screen.getByLabelText(/Lookback Period/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/Atr Multiplier/i)).toBeInTheDocument()
    })
  })

  it('renders checkboxes for boolean params', async () => {
    strategiesApi.list.mockResolvedValue({
      data: [{ name: 'InstitutionalFlowStrategy', default_config: IFS_DEFAULTS }],
    })

    render(
      <StrategyConfigForm
        strategyName="InstitutionalFlowStrategy"
        config={{}}
        onChange={() => {}}
      />
    )

    await waitFor(() => {
      expect(screen.getByLabelText(/Use Rsi Filter/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/Use Rsi Filter/i)).toHaveAttribute('type', 'checkbox')
    })
  })

  it('calls onChange with defaults on mount', async () => {
    strategiesApi.list.mockResolvedValue({
      data: [{ name: 'ATRBreakout', default_config: ATR_BREAKOUT_DEFAULTS }],
    })
    const onChange = vi.fn()

    render(<StrategyConfigForm strategyName="ATRBreakout" config={{}} onChange={onChange} />)

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith(ATR_BREAKOUT_DEFAULTS)
    })
  })

  it('calls onChange with updated value when a number input changes', async () => {
    strategiesApi.list.mockResolvedValue({
      data: [{ name: 'ATRBreakout', default_config: ATR_BREAKOUT_DEFAULTS }],
    })
    const onChange = vi.fn()

    render(
      <StrategyConfigForm
        strategyName="ATRBreakout"
        config={{ ...ATR_BREAKOUT_DEFAULTS }}
        onChange={onChange}
      />
    )

    await waitFor(() => screen.getByLabelText(/Lookback Period/i))

    // fireEvent.change is more reliable than userEvent.type for number inputs in jsdom
    fireEvent.change(screen.getByLabelText(/Lookback Period/i), {
      target: { name: 'lookback_period', value: '10', type: 'number' },
    })

    const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1][0]
    expect(lastCall.lookback_period).toBe(10)
  })

  it('calls onChange with toggled boolean when checkbox changes', async () => {
    strategiesApi.list.mockResolvedValue({
      data: [{ name: 'InstitutionalFlowStrategy', default_config: IFS_DEFAULTS }],
    })
    const onChange = vi.fn()
    const user = userEvent.setup()

    render(
      <StrategyConfigForm
        strategyName="InstitutionalFlowStrategy"
        config={{ ...IFS_DEFAULTS }}
        onChange={onChange}
      />
    )

    await waitFor(() => screen.getByLabelText(/Use Rsi Filter/i))

    await user.click(screen.getByLabelText(/Use Rsi Filter/i))

    const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1][0]
    expect(lastCall.use_rsi_filter).toBe(false)
  })

  it('shows error message when API call fails', async () => {
    strategiesApi.list.mockRejectedValue(new Error('Network error'))

    render(<StrategyConfigForm strategyName="ATRBreakout" config={{}} onChange={() => {}} />)

    await waitFor(() => {
      expect(screen.getByText(/Could not load strategy parameters/i)).toBeInTheDocument()
    })
  })

  it('resets config when strategy changes', async () => {
    strategiesApi.list
      .mockResolvedValueOnce({
        data: [{ name: 'ATRBreakout', default_config: ATR_BREAKOUT_DEFAULTS }],
      })
      .mockResolvedValueOnce({
        data: [{ name: 'InstitutionalFlowStrategy', default_config: IFS_DEFAULTS }],
      })

    const onChange = vi.fn()
    const { rerender } = render(
      <StrategyConfigForm strategyName="ATRBreakout" config={{}} onChange={onChange} />
    )

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(ATR_BREAKOUT_DEFAULTS))

    rerender(
      <StrategyConfigForm
        strategyName="InstitutionalFlowStrategy"
        config={{}}
        onChange={onChange}
      />
    )

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(IFS_DEFAULTS))
  })
})
