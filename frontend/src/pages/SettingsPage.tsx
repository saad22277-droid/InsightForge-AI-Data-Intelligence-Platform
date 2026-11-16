import { useState } from 'react'
import { Activity, Plug } from 'lucide-react'
import { api, API_URL, apiError } from '../lib/api'
import { int } from '../lib/format'
import { Badge, Busy, DataTable, ErrorNote, Panel, Stat, StatGrid } from '../components/ui'
import { useAsync } from '../hooks'

type Status = {
  settings: Record<string, string | number | boolean>
  counts: Record<string, number>
  metrics: { uptime_seconds: number; counters: Record<string, number>; latency: Record<string, { count: number; avg_ms: number; p95_ms: number }> }
}
type AiStatus = { configured: boolean; model: string; reachable: boolean | null; latency_ms?: number; error?: { kind: string; message: string }; last_error?: { kind: string; message: string } | null; free_tier_notice: string }
type Ready = { status: string; checks: Record<string, boolean> }

export default function SettingsPage() {
  const status = useAsync(async () => (await api.get<Status>('/system/status')).data, 'status')
  const ready = useAsync(async () => (await api.get<Ready>('/health/ready')).data, 'ready')
  const ai = useAsync(async () => (await api.get<AiStatus>('/ai/status')).data, 'ai')
  const [live, setLive] = useState<AiStatus | null>(null)
  const [testing, setTesting] = useState(false)
  const [error, setError] = useState('')

  const test = async () => {
    setTesting(true)
    setError('')
    try {
      setLive((await api.get<AiStatus>('/ai/status', { params: { check: true } })).data)
    } catch (err) { setError(apiError(err)) } finally { setTesting(false) }
  }

  const a = live ?? ai.data
  const s = status.data
  return (
    <>
      <Panel title="Gemini AI service" subtitle="Configured through environment variables in the backend .env file (never stored in the browser).">
        {ai.loading && <Busy />}
        {a && (
          <>
            <StatGrid>
              <Stat label="API key" value={a.configured ? 'Configured' : 'Not set'} tone={a.configured ? 'green' : 'orange'} />
              <Stat label="Model" value={a.model} tone="purple" />
              <Stat label="Connection" value={a.reachable === null ? 'Not tested' : a.reachable ? 'Working' : 'Failed'} hint={a.latency_ms ? `${a.latency_ms} ms` : undefined} tone={a.reachable ? 'green' : a.reachable === false ? 'red' : 'blue'} />
            </StatGrid>
            {a.error && <div className="alert alert-error">{a.error.message}</div>}
            {!a.configured && <div className="alert alert-warn"><span>Set <code>GEMINI_API_KEY</code> (and optionally <code>GEMINI_MODEL</code>) in <code>.env</code>, then restart the backend. Core analysis, cleaning, visualization and ML work without it.</span></div>}
            <p className="muted small">{a.free_tier_notice}</p>
            <button className="primary-button" disabled={!a.configured || testing} onClick={() => void test()}><Plug size={14} /> {testing ? 'Testing…' : 'Test Gemini connection (uses 1 request)'}</button>
          </>
        )}
        <ErrorNote message={error} />
      </Panel>

      <Panel title="System health">
        {ready.data && (
          <div className="button-row">
            <Badge tone={ready.data.status === 'ready' ? 'green' : 'red'}>{ready.data.status}</Badge>
            {Object.entries(ready.data.checks).map(([k, v]) => <Badge key={k} tone={v ? 'green' : 'red'}>{k.replace('_', ' ')}</Badge>)}
          </div>
        )}
        <p className="muted small"><Activity size={13} /> Backend: {API_URL}</p>
        {status.loading && <Busy />}
        <ErrorNote message={status.error} onRetry={status.reload} />
        {s && (
          <>
            <StatGrid>
              <Stat label="Datasets" value={int(s.counts.datasets)} tone="blue" />
              <Stat label="Model runs" value={int(s.counts.model_runs)} tone="purple" />
              <Stat label="Reports" value={int(s.counts.reports)} tone="green" />
              <Stat label="Chat messages" value={int(s.counts.chat_messages)} tone="orange" />
            </StatGrid>
            <h4>Operational metrics <small className="muted">(since backend start, {Math.round(s.metrics.uptime_seconds / 60)} min)</small></h4>
            <DataTable columns={['Counter', 'Value']} rows={Object.entries(s.metrics.counters).sort().map(([k, v]) => [k, int(v)])} empty="No activity yet." />
            <DataTable columns={['Operation', 'Count', 'Average', 'p95']} rows={Object.entries(s.metrics.latency).map(([k, v]) => [k, int(v.count), `${v.avg_ms} ms`, `${v.p95_ms} ms`])} empty="" />
          </>
        )}
      </Panel>

      <Panel title="Effective configuration" subtitle="Read-only. Edit backend/.env and restart to change. Secrets are never shown.">
        {s && <DataTable columns={['Setting', 'Value']} rows={Object.entries(s.settings).map(([k, v]) => [<code key="k">{k}</code>, String(v)])} />}
      </Panel>
    </>
  )
}
