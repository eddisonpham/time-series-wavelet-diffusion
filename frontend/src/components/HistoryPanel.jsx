import { useState, useEffect } from 'react'
import axios from 'axios'

export default function HistoryPanel() {
  const [activeTab, setActiveTab] = useState('train')
  const [trainRuns, setTrainRuns] = useState([])
  const [dataRuns, setDataRuns] = useState([])
  const [generations, setGenerations] = useState([])
  const [lightbox, setLightbox] = useState(null)

  const fetchAll = async () => {
    try {
      const [{ data: t }, { data: d }, { data: g }] = await Promise.all([
        axios.get('/api/train'),
        axios.get('/api/data'),
        axios.get('/api/generate')
      ])
      setTrainRuns(t); setDataRuns(d); setGenerations(g)
    } catch {}
  }

  useEffect(() => { fetchAll() }, [])

  const deleteRun = async (id) => {
    await axios.delete(`/api/train/${id}`)
    fetchAll()
  }

  return (
    <div>
      <div className="page-header">
        <h2>History</h2>
        <p>All pipeline runs and generated outputs</p>
      </div>

      <div className="stat-row">
        <div className="stat-chip">
          <div className="label">Data Runs</div>
          <div className="value">{dataRuns.length}</div>
        </div>
        <div className="stat-chip">
          <div className="label">Train Runs</div>
          <div className="value">{trainRuns.length}</div>
        </div>
        <div className="stat-chip">
          <div className="label">Generations</div>
          <div className="value">{generations.length}</div>
        </div>
        <div className="stat-chip">
          <div className="label">Success Rate</div>
          <div className="value">
            {trainRuns.length === 0 ? '—' :
              Math.round(trainRuns.filter(r => r.status === 'success').length / trainRuns.length * 100) + '%'}
          </div>
        </div>
      </div>

      <div className="inline-tabs">
        {['train', 'data', 'generations'].map(t => (
          <div key={t} className={`inline-tab${activeTab === t ? ' active' : ''}`}
            onClick={() => setActiveTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)} ({
              t === 'train' ? trainRuns.length :
              t === 'data' ? dataRuns.length : generations.length
            })
          </div>
        ))}
      </div>

      {activeTab === 'train' && (
        <div className="card">
          {trainRuns.length === 0
            ? <div className="empty">No training runs yet</div>
            : (
              <table className="run-table">
                <thead>
                  <tr>
                    <th>#</th><th>Date</th><th>Epochs</th><th>Img</th>
                    <th>Wavelet</th><th>Batch</th><th>LR</th><th>Status</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {trainRuns.map(r => (
                    <tr key={r.id}>
                      <td className="mono" style={{ color: 'var(--amber)' }}>#{r.id}</td>
                      <td className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        {new Date(r.created_at).toLocaleString()}
                      </td>
                      <td className="mono">{r.epochs}</td>
                      <td className="mono">{r.params.image_size}px</td>
                      <td className="mono">{r.params.wavelet}</td>
                      <td className="mono">{r.params.batch_size}</td>
                      <td className="mono">{r.params.lr}</td>
                      <td><StatusBadge status={r.status} /></td>
                      <td>
                        <button className="btn btn-danger"
                          style={{ padding: '4px 10px', fontSize: 11 }}
                          onClick={() => deleteRun(r.id)}>✕</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        </div>
      )}

      {activeTab === 'data' && (
        <div className="card">
          {dataRuns.length === 0
            ? <div className="empty">No data runs yet</div>
            : (
              <table className="run-table">
                <thead>
                  <tr><th>#</th><th>Date</th><th>Days</th><th>Rows</th><th>μ</th><th>σ</th><th>λ</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {dataRuns.map(r => (
                    <tr key={r.id}>
                      <td className="mono" style={{ color: 'var(--amber)' }}>#{r.id}</td>
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
            )}
        </div>
      )}

      {activeTab === 'generations' && (
        <div className="card">
          {generations.length === 0
            ? <div className="empty">No generations yet</div>
            : (
              <div className="gallery-grid">
                {generations.map(g => (
                  <div key={g.id} className="gallery-item" onClick={() => setLightbox(g.image_url)}>
                    <img src={g.image_url} alt={`gen-${g.id}`}
                      onError={e => { e.target.style.display = 'none' }} />
                    <div className="gallery-item-footer">
                      #{g.id} · Run #{g.train_run_id} ·{' '}
                      {new Date(g.created_at).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>
            )}
        </div>
      )}

      {lightbox && (
        <div
          style={{
            position: 'fixed', inset: 0,
            background: 'rgba(4,6,8,0.93)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 1000, cursor: 'zoom-out', backdropFilter: 'blur(6px)'
          }}
          onClick={() => setLightbox(null)}
        >
          <img src={lightbox} style={{ maxWidth: '90vw', maxHeight: '90vh', borderRadius: 10 }} />
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }) {
  const cls = status === 'success' ? 'badge-success'
    : status === 'running' ? 'badge-running'
    : status === 'failed' ? 'badge-failed' : 'badge-pending'
  return <span className={`badge ${cls}`}>{status}</span>
}