# Frontend Revamp — Design Spec
**Date:** 2026-03-30
**Scope:** Visual revamp only — same pages, same features, same API contracts. No new routes or business logic.

---

## Goal

Replace the current Bootstrap-like light theme (dark top navbar, white backgrounds) with a professional dark fintech aesthetic modelled on TradingView / Robinhood Dark. Replace the top navbar with an expanded sidebar. Preserve every existing feature, API call, and state management pattern exactly as-is.

---

## Design System

### Color Tokens

| CSS Variable | Value | Usage |
|---|---|---|
| `--bg-base` | `#0f1729` | Page background |
| `--bg-surface` | `#1a2744` | Cards, table rows, panels |
| `--bg-elevated` | `#243358` | Active nav, hover states, selected rows |
| `--bg-sidebar` | `#0a1020` | Sidebar background |
| `--border` | `#1e3050` | All borders and dividers |
| `--accent` | `#00d4a0` | Teal — positive P/L, active badges, primary buttons, active nav indicator |
| `--accent-blue` | `#4a7fb5` | Secondary labels, form hints, icon fills |
| `--text-primary` | `#e2e8f0` | All main text |
| `--text-secondary` | `#8ba3c0` | Labels, metadata, column headers |
| `--success` | `#00d4a0` | Positive P/L, "active" status |
| `--danger` | `#f85149` | Negative P/L, errors, "closed/stopped" status |
| `--warning` | `#d29922` | Paused status, caution states |

### Typography

- Font stack: `-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
- All numeric columns: `font-variant-numeric: tabular-nums` so values align cleanly
- Monospace (`"JetBrains Mono", "Fira Code", monospace`) for log entries only

### Spacing & Radius

- Base unit: `8px`
- Card border-radius: `10px`
- Button border-radius: `6px`
- Badge border-radius: `4px`
- Sidebar width: `220px` (fixed)
- Main content padding: `24px`

---

## Layout Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  SIDEBAR 220px (fixed, full height)  │  MAIN (flex-1, scroll)│
│                                       │                       │
│  ┌─ Logo block ─────────────────┐    │  [page content]       │
│  │  MOK  /  Algo Trader         │    │                       │
│  └──────────────────────────────┘    │                       │
│                                       │                       │
│  ┌─ Nav links ──────────────────┐    │                       │
│  │  ▪ Dashboard   ← active      │    │                       │
│  │    Operations                │    │                       │
│  │    Logs                      │    │                       │
│  └──────────────────────────────┘    │                       │
│                                       │                       │
│  ┌─ Footer ─────────────────────┐    │                       │
│  │  ● Connected  (status dot)   │    │                       │
│  └──────────────────────────────┘    │                       │
└──────────────────────────────────────────────────────────────┘
```

**Active nav link style:** `--bg-elevated` background + `3px solid --accent` left border strip.

**App.jsx changes:**
- Remove `<nav>` block with top navbar
- Add outer `<div class="app-layout">` with flex-row
- Add `<nav class="sidebar">` containing logo, nav links (`<NavLink>` from react-router-dom), and connection status
- Wrap existing `<Routes>` in `<main class="main-content">`

---

## Page Designs

### Dashboard

**Layout:** Stats row (4 cards) → Operations table (full width, flex-1)

**Stats cards (4):**
1. Total Equity
2. Total P/L (coloured green/red)
3. Active Operations count
4. Total Trades count

Each card: `--bg-surface` background, `--border` border, `--accent-blue` label, large bold value.

**Operations table:**
- Columns: Asset | Strategy | Bar Sizes | Status | P/L | P/L % | Action
- Rows: `--bg-surface` base, hover `--bg-elevated`
- Status badges: teal (active), yellow (paused), red (closed) — pill shape
- P/L cells: teal if positive, red if negative
- "View" action: teal text link, no button chrome

Auto-refreshes every 5s — unchanged.

---

### Operations List

**Layout:** Tab strip (filter) → table

**Status filter:** Replace dropdown with inline tab strip: `All | Active | Paused | Closed`
Active tab gets `--accent` underline. Each tab shows count badge.

**Table:** Same columns as current. Pause/Resume/Stop action buttons become icon buttons with tooltips to save space. Confirmation dialog for Stop — unchanged logic.

---

### Operation Detail

**Layout:** Page header (asset name + status badge + action buttons) → pill tab bar → tab content

**Pill tabs:** `All | Overview | Positions | Trades | Orders | Market Data`
Active pill: `--bg-elevated` + `--accent` border-bottom.

**Overview tab:** 2-column card grid for metadata (asset, strategy, bar sizes, status, initial capital, current equity, P/L, P/L %, created date). Each metric in its own labelled cell.

**Positions/Trades/Orders tabs:** Dark-themed tables, same columns as current.

**Market Data tab:** Recharts chart gets dark theme — pass `background="#1a2744"`, `stroke="--border"` for grid lines, `fill="--text-secondary"` for axis labels. No logic changes.

---

### Create Operation

**Layout:** Single-column form inside a `--bg-surface` card

**Sections:** Each logical group (Asset, Strategy, Risk Management, Recovery) becomes a labelled section with a subtle `--border` top divider and `--text-secondary` section header label.

**Inputs:** `--bg-elevated` background, `--border` border, `--text-primary` text, `--accent` focus ring. No functional changes.

**Submit button:** Teal (`--accent`) background, dark text, full-width.

---

### Logs

**Layout:** Filter bar → log list (scrollable) → sidebar stats panel

**Log entries:** Monospace font, left border strip coloured by level:
- DEBUG: `--accent-blue`
- INFO: `--text-secondary`
- WARNING: `--warning`
- ERROR / CRITICAL: `--danger`

Background of each entry: `--bg-surface` with subtle hover to `--bg-elevated`.

Filter controls: dark inputs matching the rest of the design.

Stats sidebar: level distribution counts with coloured dots.

---

## MarketDataChart Dark Theme

Pass these props into Recharts components (no chart logic changes):

```js
// CartesianGrid
stroke="#1e3050"

// XAxis / YAxis
stroke="#8ba3c0"
tick={{ fill: "#8ba3c0", fontSize: 11 }}

// Tooltip
contentStyle={{ background: "#1a2744", border: "1px solid #1e3050", borderRadius: "6px" }}
labelStyle={{ color: "#e2e8f0" }}
itemStyle={{ color: "#e2e8f0" }}

// Chart wrapper background
style={{ background: "#1a2744" }}
```

---

## Files Changed

| File | Type of change |
|---|---|
| `src/index.css` | Full replacement — CSS variables, reset, global dark styles |
| `src/App.css` | Full replacement — sidebar layout, nav link styles, main content |
| `src/App.jsx` | Structural refactor — top navbar → sidebar, layout wrappers |
| `src/pages/Dashboard.jsx` | Layout update — stats row, table styles |
| `src/pages/Operations.jsx` | Layout update — tab strip filter, table styles |
| `src/pages/OperationDetail.jsx` | Layout update — pill tabs, overview grid |
| `src/pages/CreateOperation.jsx` | Layout update — dark form sections |
| `src/pages/Logs.jsx` | Layout update — monospace entries, level borders |
| `src/components/MarketDataChart.jsx` | Dark theme props for Recharts |

**Not changed:** `src/api/client.js`, `src/utils/formatters.js`, `src/components/StrategyConfigForm.jsx`, all Zustand stores, all business logic.

---

## Testing

**Existing test:** `StrategyConfigForm.test.jsx` — must continue to pass with no modifications.

**New test:** `src/App.test.jsx`
- Render `<App>` with a `MemoryRouter`
- Assert sidebar nav links for "Dashboard", "Operations", "Logs" are present in the DOM
- Assert no top navbar `<nav>` with old class names exists

This guards against navigation regression during the CSS/JSX refactor.

---

## Out of Scope

- No new pages or routes
- No new API endpoints consumed
- No walk-forward results page (deferred)
- No performance monitor status panel (deferred)
- No Tailwind or component library introduction
- No mobile/responsive redesign
