// src/App.test.jsx
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, it, expect } from 'vitest'
import App from './App'

describe('App sidebar navigation', () => {
  it('renders sidebar nav links for Dashboard, Operations, and Logs', () => {
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    )
    expect(screen.getByRole('link', { name: /dashboard/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /operations/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /logs/i })).toBeInTheDocument()
  })

  it('does not render old top navbar with class navbar', () => {
    const { container } = render(
      <MemoryRouter>
        <App />
      </MemoryRouter>
    )
    expect(container.querySelector('.navbar')).toBeNull()
  })
})
