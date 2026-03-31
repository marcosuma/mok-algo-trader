import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Operations from './pages/Operations'
import OperationDetail from './pages/OperationDetail'
import CreateOperation from './pages/CreateOperation'
import Logs from './pages/Logs'
import './App.css'

function App() {
  return (
    <div className="app-layout">
      <nav className="sidebar">
        <div className="sidebar-logo">
          <span className="sidebar-logo-mok">MOK</span>
          <span className="sidebar-logo-sub">Algo Trader</span>
        </div>

        <div className="sidebar-nav">
          <NavLink to="/" end className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Dashboard
          </NavLink>
          <NavLink to="/operations" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Operations
          </NavLink>
          <NavLink to="/logs" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            Logs
          </NavLink>
        </div>

        <div className="sidebar-footer">
          <span className="status-dot"></span>
          <span className="status-label">Connected</span>
        </div>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/operations" element={<Operations />} />
          <Route path="/operations/create" element={<CreateOperation />} />
          <Route path="/operations/:id" element={<OperationDetail />} />
          <Route path="/logs" element={<Logs />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
