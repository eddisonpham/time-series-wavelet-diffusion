import { useState, useEffect } from 'react'
import axios from 'axios'

const DEFAULTS = {
  days: 30, mu: 0.05, sigma: 0.2, lam: 0.1,
  jump_mean: -0.02, jump_std: 0.1, year: 2022, month: 1
}

export default function DataPanel() {
  const [params, setParams] = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [runs, setRuns] = useState([])

  const fetchRuns = async () => {
    try {
      const { data } = await axios.get('/api/data')
      setRuns(data)
    } catch {
      console.error('Error fetching data runs')
    }
  }

  useEffect(() => { fetchRuns() }, [])

  const handleGenerate = async () => {
    setLoading(true); setError(null); setResult(null)
    try {
      const { data } = await axios.post('/api/data', params)
      setResult(data)
      fetchRuns()
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally { setLoading(false) }
  }

  const set = (k, v) => setParams(p => ({ ...p, [k]: v }))

  return (
    <div>
      <div className="page-header">
        <h2>Data Generation</h2>
        <p>Generate synthetic financial time-series using Merton Jump Diffusion</p>
      </div>

      <div className="stat-row">
        <div className="stat-chip">
          <div className="label">Total Runs</div>
          <div className="value">{runs.length}</div>
        </div>
        <div className="stat-chip">
          <div className="label">Last Rows</div>
          <div className="value">{runs[0]?.rows?.toLocaleString() || '—'}</div>
        </div>
        <div className="stat-chip">
          <div className="label">Output</div>
          <div className="value" style={{ fontSize: 12 }}>data/index_time_series.csv</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">MJD Parameters</div>

        <div className="grid-3" style={{ marginBottom: 14 }}>
          <div className="field">
            <label>Duration (days)</label>
            <input type="number" value={params.days} min={1} max={365}
              onChange={e => set('days', +e.target.value)} />
          </div>
          <div className="field">
            <label>Year</label>
            <input type="number" value={params.year}
              onChange={e => set('year', +e.target.value)} />
          </div>
          <div className="field">
            <label>Month</label>
            <input type="number" value={params.month} min={1} max={12}
              onChange={e => set('month', +e.target.value)} />
          </div>
        </div>

        <div className="grid-2" style={{ marginBottom: 14 }}>
          <SliderField label="Drift (μ)" k="mu" min={-0.5} max={0.5} step={0.01}
            value={params.mu} set={set} />
          <SliderField label="Volatility (σ)" k="sigma" min={0.01} max={1} step={0.01}
            value={params.sigma} set={set} />
          <SliderField label="Jump Intensity (λ)" k="lam" min={0} max={1} step={0.01}
            value={params.lam} set={set} />
          <SliderField label="Jump Mean (m)" k="jump_mean" min={-0.5} max={0.5} step={0.01}
            value={params.jump_mean} set={set} />
          <SliderField label="Jump Std (v)" k="jump_std" min={0.01} max={0.5} step={0.01}
            value={params.jump_std} set={set} />
        </div>

        {error && <div className="alert alert-error">{error}</div>}
        {result && (
          <div className="alert alert-success">
            ✓ Generated {result.rows?.toLocaleString()} rows → data/index_time_series.csv
          </div>
        )}

        <button className="btn btn-primary" onClick={handleGenerate} disabled={loading}>
          {loading ? <><div className="spinner" /> Generating…</> : <>▶ Generate Data</>}
        </button>
      </div>

      {runs.length > 0 && (
        <div className="card">
          <div className="card-title">Run History</div>
          <table className="run-table">
            <thead>
              <tr>
                <th>#</th><th>Created</th><th>Days</th><th>Rows</th>
                <th>μ</th><th>σ</th><th>λ</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map(r => (
                <tr key={r.id}>
                  <td className="mono" style={{ color: 'var(--text-dim)' }}>{r.id}</td>
                  <td className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                  <td className="mono">{r.params.days}</td>
                  <td className="mono">{r.rows?.toLocaleString()}</td>
                  <td className="mono">{r.params.mu}</td>
                  <td className="mono">{r.params.sigma}</td>
                  <td className="mono">{r.params.lam}</td>
                  <td><StatusBadge status={r.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function SliderField({ label, k, min, max, step, value, set }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="range-row">
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => set(k, parseFloat(e.target.value))} />
        <span className="range-val">{value}</span>
      </div>
    </div>
  )
}

function StatusBadge({ status }) {
  const cls = status === 'success' ? 'badge-success'
    : status === 'running' ? 'badge-running'
    : status === 'failed' ? 'badge-failed' : 'badge-pending'
  return <span className={`badge ${cls}`}>{status}</span>
}