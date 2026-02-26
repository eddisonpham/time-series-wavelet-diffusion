import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const DEFAULTS = {
  window_size: 256, stride: 64, wavelet: 'morl',
  scales: 128, image_size: 128, batch_size: 16,
  epochs: 10, lr: 0.0001, num_timesteps: 1000
}

const WAVELETS = ['morl', 'cmor', 'fbsp', 'mexh', 'gaus1', 'gaus2', 'cgau1']

export default function TrainPanel() {
  const [params, setParams] = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [runId, setRunId] = useState(null)
  const [logs, setLogs] = useState([])
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [runs, setRuns] = useState([])
  const logsRef = useRef(null)
  const evsRef = useRef(null)

  const fetchRuns = async () => {
    try { const { data } = await axios.get('/api/train'); setRuns(data) } catch {}
  }

  useEffect(() => { fetchRuns() }, [])

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [logs])

  const startTraining = async () => {
    setLoading(true); setError(null); setLogs([]); setStatus('pending')
    try {
      const { data } = await axios.post('/api/train', params)
      const id = data.id
      setRunId(id)

      // Stream logs via SSE
      if (evsRef.current) evsRef.current.close()
      const evs = new EventSource(`/api/train/${id}/logs`)
      evsRef.current = evs

      evs.onmessage = (e) => {
        const payload = JSON.parse(e.data)
        if (payload.line !== undefined) {
          setLogs(l => [...l, payload.line])
        }
        if (payload.done) {
          setStatus(payload.status)
          setLoading(false)
          evs.close()
          fetchRuns()
        }
      }
      evs.onerror = () => {
        setLoading(false)
        evs.close()
        fetchRuns()
      }
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setLoading(false)
      setStatus('failed')
    }
  }

  const set = (k, v) => setParams(p => ({ ...p, [k]: v }))

  return (
    <div>
      <div className="page-header">
        <h2>Model Training</h2>
        <p>Configure hyperparameters and train a DDPM diffusion model</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Config card */}
        <div>
          <div className="card">
            <div className="card-title">Wavelet Config</div>
            <div className="grid-2" style={{ marginBottom: 14 }}>
              <div className="field">
                <label>Wavelet</label>
                <select value={params.wavelet} onChange={e => set('wavelet', e.target.value)}>
                  {WAVELETS.map(w => <option key={w} value={w}>{w}</option>)}
                </select>
              </div>
              <IntField label="Scales" k="scales" min={16} max={512} value={params.scales} set={set} />
              <IntField label="Image Size (px)" k="image_size" min={32} max={512} value={params.image_size} set={set} />
              <IntField label="Window Size" k="window_size" min={64} max={1024} value={params.window_size} set={set} />
              <IntField label="Stride" k="stride" min={1} max={512} value={params.stride} set={set} />
            </div>
          </div>

          <div className="card">
            <div className="card-title">Training Config</div>
            <div className="grid-2">
              <IntField label="Epochs" k="epochs" min={1} max={200} value={params.epochs} set={set} />
              <IntField label="Batch Size" k="batch_size" min={1} max={128} value={params.batch_size} set={set} />
              <IntField label="Timesteps" k="num_timesteps" min={100} max={4000} value={params.num_timesteps} set={set} />
              <div className="field">
                <label>Learning Rate</label>
                <input type="number" step="0.00001" value={params.lr}
                  onChange={e => set('lr', parseFloat(e.target.value))} />
              </div>
            </div>
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          <button className="btn btn-primary" style={{ width: '100%' }}
            onClick={startTraining} disabled={loading}>
            {loading
              ? <><div className="spinner" /> Training (Run #{runId})…</>
              : <>▶ Start Training</>}
          </button>

          {status && !loading && (
            <div className={`alert ${status === 'success' ? 'alert-success' : 'alert-error'}`}
              style={{ marginTop: 10 }}>
              {status === 'success' ? `✓ Training complete — Run #${runId}` : `✗ Training failed`}
            </div>
          )}
        </div>

        {/* Logs + run list */}
        <div>
          <div className="card" style={{ height: 340 }}>
            <div className="card-title">Training Log</div>
            <div className="terminal" ref={logsRef} style={{ height: 260 }}>
              {logs.length === 0
                ? <span style={{ color: 'var(--text-muted)' }}>Waiting for training to start…</span>
                : logs.map((line, i) => (
                  <div key={i} className={`log-line ${classifyLine(line)}`}>{line}</div>
                ))}
            </div>
          </div>

          {runs.length > 0 && (
            <div className="card">
              <div className="card-title">Past Runs</div>
              <table className="run-table">
                <thead>
                  <tr><th>#</th><th>Date</th><th>Epochs</th><th>img</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {runs.slice(0, 8).map(r => (
                    <tr key={r.id}>
                      <td className="mono" style={{ color: 'var(--amber)' }}>#{r.id}</td>
                      <td className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        {new Date(r.created_at).toLocaleString()}
                      </td>
                      <td className="mono">{r.epochs}</td>
                      <td className="mono">{r.params.image_size}px</td>
                      <td><StatusBadge status={r.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function IntField({ label, k, min, max, value, set }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" min={min} max={max} value={value}
        onChange={e => set(k, parseInt(e.target.value))} />
    </div>
  )
}

function classifyLine(line) {
  if (line.includes('error') || line.includes('Error') || line.includes('failed')) return 'err'
  if (line.includes('Epoch') || line.includes('loss=')) return 'warn'
  return 'info'
}

function StatusBadge({ status }) {
  const cls = status === 'success' ? 'badge-success'
    : status === 'running' ? 'badge-running'
    : status === 'failed' ? 'badge-failed' : 'badge-pending'
  return <span className={`badge ${cls}`}>{status}</span>
}