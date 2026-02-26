import { useState, useEffect } from 'react'
import axios from 'axios'

export default function GeneratePanel() {
  const [trainRuns, setTrainRuns] = useState([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [loading, setLoading] = useState(false)
  const [image, setImage] = useState(null)
  const [genId, setGenId] = useState(null)
  const [error, setError] = useState(null)
  const [gallery, setGallery] = useState([])
  const [lightbox, setLightbox] = useState(null)

  const fetchData = async () => {
    try {
      const [{ data: trains }, { data: gens }] = await Promise.all([
        axios.get('/api/train'),
        axios.get('/api/generate')
      ])
      const successful = trains.filter(r => r.status === 'success')
      setTrainRuns(successful)
      if (successful.length > 0 && !selectedRunId) {
        setSelectedRunId(successful[0].id)
      }
      setGallery(gens)
    } catch {}
  }

  useEffect(() => { fetchData() }, [])

  const handleGenerate = async () => {
    if (!selectedRunId) return
    setLoading(true); setError(null); setImage(null)
    try {
      const { data } = await axios.post('/api/generate', { train_run_id: parseInt(selectedRunId) })
      setImage(data.image_url)
      setGenId(data.id)
      fetchData()
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally { setLoading(false) }
  }

  const selectedRun = trainRuns.find(r => r.id === parseInt(selectedRunId))

  return (
    <div>
      <div className="page-header">
        <h2>Scalogram Generation</h2>
        <p>Sample from the trained DDPM model and visualize wavelet scalograms</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16 }}>
        {/* Controls */}
        <div>
          <div className="card">
            <div className="card-title">Run Selection</div>
            {trainRuns.length === 0
              ? <div className="empty">No successful training runs found.<br/>Train a model first.</div>
              : (
                <>
                  <div className="field" style={{ marginBottom: 16 }}>
                    <label>Training Run</label>
                    <select value={selectedRunId}
                      onChange={e => setSelectedRunId(e.target.value)}>
                      {trainRuns.map(r => (
                        <option key={r.id} value={r.id}>
                          Run #{r.id} — {new Date(r.created_at).toLocaleDateString()} ({r.epochs} epochs)
                        </option>
                      ))}
                    </select>
                  </div>

                  {selectedRun && (
                    <div style={{ marginBottom: 16 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 6, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Config</div>
                      {Object.entries(selectedRun.params).map(([k, v]) => (
                        <span key={k} className="param-tag">{k}=<span>{v}</span></span>
                      ))}
                    </div>
                  )}

                  {error && <div className="alert alert-error" style={{ marginBottom: 12 }}>{error}</div>}

                  <button className="btn btn-primary" style={{ width: '100%' }}
                    onClick={handleGenerate} disabled={loading || !selectedRunId}>
                    {loading ? <><div className="spinner" /> Denoising…</> : <>⟴ Generate Scalogram</>}
                  </button>
                </>
              )}
          </div>

          {gallery.length > 0 && (
            <div className="card">
              <div className="card-title">Generated ({gallery.length})</div>
              <div className="gallery-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                {gallery.slice(0, 8).map(g => (
                  <div key={g.id} className="gallery-item"
                    onClick={() => setLightbox(g.image_url)}
                    style={genId === g.id ? { borderColor: 'var(--amber)' } : {}}>
                    <img src={g.image_url} alt={`gen-${g.id}`}
                      onError={e => { e.target.style.display = 'none' }} />
                    <div className="gallery-item-footer">#{g.id} · Run {g.train_run_id}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Scalogram viewer */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-title">Output Scalogram</div>
          <div className="scalogram-container" style={{ flex: 1, minHeight: 400 }}>
            {loading && (
              <div className="scalogram-placeholder">
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
                  <DiffusionSpinner />
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--amber)' }}>
                    Running reverse diffusion…
                  </div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)' }}>
                    {selectedRun?.params?.num_timesteps || 1000} denoising steps
                  </div>
                </div>
              </div>
            )}
            {!loading && !image && (
              <div className="scalogram-placeholder">
                <WaveIcon />
                <div style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
                  Select a run and click generate
                </div>
              </div>
            )}
            {!loading && image && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: 16, width: '100%' }}>
                <img src={image} alt="Generated scalogram"
                  style={{ maxWidth: '100%', borderRadius: 8, cursor: 'zoom-in' }}
                  onClick={() => setLightbox(image)} />
                <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-dim)' }}>
                  Generation #{genId} · Click to expand
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Lightbox */}
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

function DiffusionSpinner() {
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" style={{ animation: 'spin 1.5s linear infinite' }}>
      <circle cx="32" cy="32" r="28" fill="none" stroke="var(--border)" strokeWidth="3" />
      <circle cx="32" cy="32" r="28" fill="none" stroke="var(--amber)" strokeWidth="3"
        strokeDasharray="44 132" strokeLinecap="round" />
    </svg>
  )
}

function WaveIcon() {
  return (
    <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" opacity="0.25">
      <path d="M2 12h2l3-7 4 14 3-10 2 3h6" />
    </svg>
  )
}