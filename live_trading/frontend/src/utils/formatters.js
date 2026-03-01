export const formatCurrency = (value) => {
    if (value === null || value === undefined) return '-'
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value)
}

// Format a market price with appropriate precision.
// Forex rates (< 100) get 5 decimal places; larger prices (crypto, indices) get
// 2 decimal places with comma grouping. No currency symbol — prices are rates,
// not dollar amounts.
export const formatForexPrice = (value) => {
    if (value === null || value === undefined) return '-'
    const abs = Math.abs(value)
    if (abs >= 100) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        }).format(value)
    }
    return value.toFixed(5)
}

export const formatPercent = (value) => {
    if (value === null || value === undefined) return '-'
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
}

export const formatDate = (dateString) => {
    if (!dateString) return '-'
    return new Date(dateString).toLocaleString()
}

