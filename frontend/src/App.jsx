import { useState } from 'react'
import DataPanel from './components/DataPanel'
import TrainPanel from './components/TrainPanel'
import GeneratePanel from './components/GeneratePanel'
import HistoryPanel from './components/HistoryPanel'

const NAV = [
  { id: 'data',     label: 'Data',      icon: DatabaseIcon,    desc: 'Synthetic data generation' },
  { id: 'train',    label: 'Train',     icon: CpuIcon,         desc: 'Train diffusion model' },
  { id: 'generate', label: 'Generate',  icon: WaveformIcon,    desc: 'Synthesize scalograms' },
  { id: 'history',  label: 'History',   icon: ClockIcon,       desc: 'All runs & outputs' },
]

export default function App() {
  const [tab, setTab] = useState('data')

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>WaveDiff</h1>
          <p>Wavelet Diffusion Studio</p>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">Pipeline</div>
          {NAV.map(n => (
            <div
              key={n.id}
              className={`nav-item${tab === n.id ? ' active' : ''}`}
              onClick={() => setTab(n.id)}
            >
              <n.icon />
              {n.label}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>API connected</span>
        </div>
      </aside>

      <main className="main">
        {tab === 'data'     && <DataPanel />}
        {tab === 'train'    && <TrainPanel />}
        {tab === 'generate' && <GeneratePanel />}
        {tab === 'history'  && <HistoryPanel />}
      </main>
    </div>
  )
}

// ── Inline SVG icons ────────────────────────────────────────────────────────

function DatabaseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4.03 3-9 3S3 13.66 3 12" />
      <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
    </svg>
  )
}

function CpuIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
      <line x1="1" y1="9" x2="4" y2="9" />
      <line x1="1" y1="15" x2="4" y2="15" />
      <line x1="20" y1="9" x2="23" y2="9" />
      <line x1="20" y1="15" x2="23" y2="15" />
    </svg>
  )
}

function WaveformIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M2 12h2l3-7 4 14 3-10 2 3h6" />
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 6v6l4 2" />
    </svg>
  )
}