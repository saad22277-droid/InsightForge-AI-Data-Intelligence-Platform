import { useState } from 'react'
import { Download, FileText, Trash2 } from 'lucide-react'
import { api, apiError, downloadUrl } from '../lib/api'
import { bytes, when } from '../lib/format'
import type { MlRun, PageCtx, ReportMeta } from '../types'
import { Badge, Busy, DataTable, Empty, ErrorNote, Field, Panel } from '../components/ui'
import { useAsync } from '../hooks'

export default function ReportsPage({ datasetId, filename, notify, fail }: PageCtx) {
  const reports = useAsync(async () => (await api.get<ReportMeta[]>(`/datasets/${datasetId}/reports`)).data, datasetId)
  const runs = useAsync(async () => (await api.get<MlRun[]>(`/datasets/${datasetId}/ml/runs`)).data, datasetId)
  const ai = useAsync(async () => (await api.get<{ configured: boolean }>('/ai/status')).data, 'ai')

  const [title, setTitle] = useState('')
  const [includeAi, setIncludeAi] = useState(false)
  const [picked, setPicked] = useState<number[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const generate = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await api.post<ReportMeta>(`/datasets/${datasetId}/reports`, {
        title: title || undefined,
        include_ai: includeAi,
        run_ids: picked.length ? picked : undefined,
      })
      const notes = r.data.sections?.ai_notes ?? []
      notify(includeAi && !r.data.ai_included
        ? `Report created without AI explanations. ${notes[0] ?? ''}`
        : 'Report generated.')
      reports.reload()
    } catch (err) {
      setError(apiError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Panel title="Generate analytical report" subtitle={`A professional PDF built from the real results for ${filename}: overview, quality, EDA, charts, models, recommendations and limitations.`}>
        <div className="form-grid">
          <Field label="Report title (optional)"><input value={title} maxLength={150} onChange={(e) => setTitle(e.target.value)} placeholder={`Data Intelligence Report - ${filename.replace(/\.csv$/i, '')}`} /></Field>
          <Field label="Gemini explanations" hint={ai.data && !ai.data.configured ? 'Gemini is not configured; the report is created without AI text.' : 'Uses up to 5 Gemini requests from your quota.'}>
            <label className="check-inline"><input type="checkbox" checked={includeAi} onChange={(e) => setIncludeAi(e.target.checked)} /> Include AI-written explanations</label>
          </Field>
        </div>
        {(runs.data ?? []).length > 0 && (
          <details className="features-box">
            <summary>Models to include: {picked.length ? `${picked.length} selected` : 'best model per task (default)'}</summary>
            <div className="check-grid">
              {(runs.data ?? []).map((r) => (
                <label key={r.id}><input type="checkbox" checked={picked.includes(r.id)} onChange={(e) => setPicked(e.target.checked ? [...picked, r.id].slice(-5) : picked.filter((i) => i !== r.id))} /> {r.model_name} v{r.version} <small className="muted">{r.primary_metric} {r.primary_metric_value?.toFixed(3)}</small></label>
              ))}
            </div>
          </details>
        )}
        <ErrorNote message={error} />
        <div className="button-row">
          <button className="primary-button" onClick={() => void generate()} disabled={busy}><FileText size={15} /> {busy ? 'Generating…' : 'Generate PDF report'}</button>
          <a className="ghost-button" href={downloadUrl(`/datasets/${datasetId}/exports/columns.csv`)}><Download size={14} /> Column profile (CSV)</a>
        </div>
        {busy && <Busy text="Building charts and PDF… this can take a few seconds (longer with AI)." />}
      </Panel>

      <Panel title="Generated reports">
        {reports.loading ? <Busy /> : (reports.data ?? []).length === 0 ? (
          <Empty icon={<FileText size={24} />} title="No reports yet" text="Generate your first report above." />
        ) : (
          <DataTable
            columns={['Title', 'Created', 'Size', 'AI', '']}
            rows={(reports.data ?? []).map((r) => [
              <strong key="t">{r.title}</strong>, when(r.created_at), bytes(r.size_bytes),
              r.ai_included ? <Badge key="a" tone="green">Gemini</Badge> : <Badge key="a">none</Badge>,
              <div key="x" className="button-row">
                <a className="primary-button small" href={downloadUrl(r.download_url)} target="_blank" rel="noreferrer"><Download size={13} /> PDF</a>
                <button className="icon-button small" aria-label="Delete report" onClick={async () => {
                  if (!window.confirm('Delete this report?')) return
                  try { await api.delete(`/reports/${r.id}`); reports.reload() } catch (e) { fail(apiError(e)) }
                }}><Trash2 size={14} /></button>
              </div>,
            ])}
          />
        )}
      </Panel>
    </>
  )
}
