import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Export the base URL for direct fetch usage
export const API_BASE = API_BASE_URL

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
})

// Strategies
export const strategiesApi = {
    list: () => apiClient.get('/api/strategies'),
}

// Operations
export const operationsApi = {
    list: (status) => {
        const params = status ? { status } : {}
        return apiClient.get('/api/operations', { params })
    },

    get: (id) => apiClient.get(`/api/operations/${id}`),

    create: (data) => apiClient.post('/api/operations', data),

    delete: (id) => apiClient.delete(`/api/operations/${id}`),

    pause: (id) => apiClient.post(`/api/operations/${id}/pause`),

    resume: (id) => apiClient.post(`/api/operations/${id}/resume`),
}

// Positions
export const positionsApi = {
    list: (operationId) => apiClient.get(`/api/operations/${operationId}/positions`),
}

// Trades
export const tradesApi = {
    list: (operationId) => apiClient.get(`/api/operations/${operationId}/trades`),
}

// Orders
export const ordersApi = {
    list: (operationId) => apiClient.get(`/api/operations/${operationId}/orders`),
}

// Market Data
export const marketDataApi = {
    list: (operationId, barSize = null, limit = 1000) => {
        const params = { limit }
        if (barSize) params.bar_size = barSize
        return apiClient.get(`/api/operations/${operationId}/market-data`, { params })
    },
    count: (operationId) => apiClient.get(`/api/operations/${operationId}/market-data/count`),
}

// Statistics
export const statsApi = {
    operation: (operationId) => apiClient.get(`/api/operations/${operationId}/stats`),
    overall: () => apiClient.get('/api/stats/overall'),
}

export default apiClient

