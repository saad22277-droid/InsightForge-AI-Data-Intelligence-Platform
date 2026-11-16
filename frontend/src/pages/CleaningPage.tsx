import { useState } from 'react'
import { Download, Eye, History, Plus, RotateCcw, Trash2, Wand2 } from 'lucide-react'
import { api, apiError, downloadUrl } from '../lib/api'
import { int, num, when } from '../lib/format'
import type { CleaningPreview, Metadata, PageCtx, Recommendation } from '../types'
import { Badge, Busy, DataTable, Empty, ErrorNote, Field, Panel, Stat, StatGrid } from '../components/ui'
import { useAsync } from '../hooks'

type Op = Record<string, unknown> & { type: string }
type QueueItem = { key: string; label: string; operation: Op }
type VersionRow = { version_number: number; row_count?: number; column_count?: number; created_at?: string }
type HistoryRow = { id: number; source_version: number | null; result_version: number; operations: Op[]; summary: { note?: string | null; delta?: Record<string, number> }; created_at: string }

const OP_TYPES = [
  'fill_missing', 'drop_duplicates', 'drop_missing_rows', 'drop_columns', 'convert_dtype',
  'handle_outliers', 'standardize_text', 'replace_invalid', 'invalid_to_missing',
  'scale_column', 'normalize_column_names',
] as const

const NO_COLUMN = new Set(['drop_duplicates', 'drop_missing_rows', 'normalize_column_names'])

function describe(op: Op): string {
  const col = (op.column ?? (Array.isArray(op.columns) ? (op.columns as string[]).join(', ') : '')) as string
  const extra = op.strategy ?? op.dtype ?? op.action ?? op.method ?? ''
  return `${op.type.replace(/_/g, ' ')}${col ? ` · ${col}` : ''}${extra ? ` (${extra})` : ''}`
}

export default function CleaningPage({ datasetId, notify, fail, refresh }: PageCtx) {
  const recs = useAsync(
    async () => (await api.get<{ recommendations: Recommendation[]; recommendation_count: number }>(`/datasets/${datasetId}/cleaning/recommendations`)).data,
    datasetId,
  )
  const meta = useAsync(async () => (await api.get<Metadata>(`/datasets/${datasetId}/metadata`)).data, datasetId)
  const versions = useAsync(async () => (await api.get<{ versions: VersionRow[] }>(`/datasets/${datasetId}/versions`)).data.versions, datasetId)
  const history = useAsync(async () => (await api.get<HistoryRow[]>(`/datasets/${datasetId}/cleaning/history`)).data, datasetId)

  const [queue, setQueue] = useState<QueueItem[]>([])
  const [preview, setPreview] = useState<CleaningPreview | null>(null)
  const [busy, setBusy] = useState('')
  const [previewError, setPreviewError] = useState('')
  const [note, setNote] = useState('')

  const addOp = (label: string, operation: Op) => {
    const key = JSON.stringify(operation)
    setQueue((q) => (q.some((i) => i.key === key) ? q : [...q, { key, label, operation }]))
    setPreview(null)
  }
  const removeOp = (key: string) => {
    setQueue((q) => q.filter((i) => i.key !== key))
    setPreview(null)
  }
  const ops = queue.map((i) => i.operation)

  const runPreview = async () => {
    setBusy('preview')
    setPreviewError('')
    try {
      setPreview((await api.post<CleaningPreview>(`/datasets/${datasetId}/cleaning/preview`, { operations: ops })).data)
    } catch (err) {
      setPreview(null)
      setPreviewError(apiError(err))
    } finally {
      setBusy('')
    }
  }

  const apply = async () => {
    if (!window.confirm('Apply these operations and save the result as a new dataset version? Previous versions are kept.')) return
    setBusy('apply')
    setPreviewError('')
    try {
      const r = await api.post<{ new_version: number }>(`/datasets/${datasetId}/cleaning/apply`, { operations: ops, note: note || undefined })
      notify(`Cleaning applied. Dataset version ${r.data.new_version} created; the original is preserved.`)
      setQueue([])
      setPreview(null)
      setNote('')
      recs.reload(); meta.reload(); versions.reload(); history.reload(); refresh()
    } catch (err) {
      setPreviewError(apiError(err))
    } finally {
      setBusy('')
    }
  }

  const activate = async (v: number) => {
    try {
      await api.post(`/datasets/${datasetId}/versions/${v}/activate`)
      notify(`Version ${v} is now the active dataset used by analysis, ML and AI.`)
      recs.reload(); meta.reload(); refresh()
    } catch (err) {
      fail(apiError(err))
    }
  }

  const recList = recs.data?.recommendations ?? []
  const active = meta.data?.active_version

  return (
    <>
      <Panel
        title="Cleaning recommendations"
        subtitle="Detected from your data. Nothing changes until you queue, preview and apply."
        actions={recList.length > 0 && (
          <button className="ghost-button" onClick={() => recList.forEach((r) => addOp(r.title, r.operation as Op))}>
            <Plus size={14} /> Queue all
          </button>
        )}
      >
        {recs.loading && <Busy text="Analysing data quality…" />}
        <ErrorNote message={recs.error} onRetry={recs.reload} />
        {!recs.loading && !recs.error && recList.length === 0 && (
          <Empty icon={<Wand2 size={24} />} title="Nothing to clean" text="No missing values, duplicates, invalid values or formatting problems were detected." />
        )}
        <div className="rec-list">
          {recList.map((r) => {
            const queued = queue.some((i) => i.key === JSON.stringify(r.operation))
            return (
              <div key={r.id} className="rec-item">
                <div>
                  <div className="rec-title">
                    <Badge tone={r.severity === 'warning' ? 'orange' : 'blue'}>{r.category.replace('_', ' ')}</Badge>
                    <strong>{r.title}</strong>
                  </div>
                  <p>{r.reason}</p>
                  {r.estimated_impact && <small className="muted">Impact: {r.estimated_impact}</small>}
                </div>
                <button className="ghost-button" disabled={queued} onClick={() => addOp(r.title, r.operation as Op)}>
                  {queued ? 'Queued' : <><Plus size={14} /> Queue</>}
                </button>
              </div>
            )
          })}
        </div>
      </Panel>

      <ManualBuilder columns={meta.data?.columns.map((c) => c.name) ?? []} onAdd={addOp} />

      <Panel
        title={`Operation queue (${queue.length})`}
        subtitle="Operations run in order on a copy of the active version."
        actions={
          <div className="button-row">
            <button className="ghost-button" disabled={queue.length === 0 || !!busy} onClick={() => void runPreview()}>
              <Eye size={14} /> {busy === 'preview' ? 'Previewing…' : 'Preview'}
            </button>
            <button className="primary-button" disabled={queue.length === 0 || !!busy} onClick={() => void apply()}>
              {busy === 'apply' ? 'Applying…' : 'Apply as new version'}
            </button>
          </div>
        }
      >
        {queue.length === 0 ? (
          <p className="muted pad">Queue recommendations above or build an operation manually.</p>
        ) : (
          <>
            <ol className="queue">
              {queue.map((i) => (
                <li key={i.key}>
                  <span>{i.label} <small className="muted">— {describe(i.operation)}</small></span>
                  <button className="icon-button small" onClick={() => removeOp(i.key)} aria-label="Remove operation"><Trash2 size={14} /></button>
                </li>
              ))}
            </ol>
            <Field label="Note (optional)"><input value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} placeholder="Why are you applying these changes?" /></Field>
          </>
        )}
        <ErrorNote message={previewError} />
        {busy === 'preview' && <Busy text="Computing before/after comparison…" />}
        {preview && <PreviewView preview={preview} />}
      </Panel>

      <Panel title="Dataset versions" subtitle="Every cleaning run creates a new version. The original upload is never modified.">
        <DataTable
          columns={['Version', 'Rows', 'Columns', 'Created', 'Status', 'Actions']}
          rows={(versions.data ?? []).slice().sort((a, b) => b.version_number - a.version_number).map((v) => [
            `v${v.version_number}`, int(v.row_count), int(v.column_count), when(v.created_at),
            v.version_number === active ? <Badge key="a" tone="green">active</Badge> : <Badge key="i">inactive</Badge>,
            <div key="x" className="button-row">
              <a className="ghost-button" href={downloadUrl(`/datasets/${datasetId}/versions/${v.version_number}/download`)}><Download size={13} /> CSV</a>
              {v.version_number !== active && <button className="ghost-button" onClick={() => void activate(v.version_number)}><RotateCcw size={13} /> Make active</button>}
            </div>,
          ])}
        />
      </Panel>

      <Panel title="Cleaning history" subtitle="Audit trail of applied operations">
        {(history.data ?? []).length === 0 ? <p className="muted pad"><History size={14} /> No cleaning has been applied yet.</p> : (
          <DataTable
            columns={['Result', 'From', 'Operations', 'Net change', 'Note', 'When']}
            rows={(history.data ?? []).map((h) => [
              `v${h.result_version}`, h.source_version ? `v${h.source_version}` : '—',
              <span key="o" className="wrap">{h.operations.map(describe).join('; ')}</span>,
              h.summary.delta ? `${h.summary.delta.rows >= 0 ? '+' : ''}${h.summary.delta.rows} rows, ${h.summary.delta.missing_cells >= 0 ? '+' : ''}${h.summary.delta.missing_cells} missing` : '—',
              h.summary.note ?? '—', when(h.created_at),
            ])}
          />
        )}
      </Panel>
    </>
  )
}

function PreviewView({ preview }: { preview: CleaningPreview }) {
  const d = preview.delta
  const sign = (n: number) => (n > 0 ? `+${int(n)}` : int(n))
  const cols = (rows: Record<string, unknown>[]) => (rows[0] ? Object.keys(rows[0]).slice(0, 8) : [])
  const cell = (v: unknown) => (v === null || v === undefined ? <span className="muted">null</span> : String(v))
  return (
    <div className="preview">
      <h4>Before → after</h4>
      <StatGrid>
        <Stat label="Rows" value={`${int(preview.before.rows)} → ${int(preview.after.rows)}`} hint={sign(d.rows)} tone="blue" />
        <Stat label="Columns" value={`${preview.before.columns} → ${preview.after.columns}`} hint={sign(d.columns)} tone="purple" />
        <Stat label="Missing cells" value={`${int(preview.before.missing_cells)} → ${int(preview.after.missing_cells)}`} hint={sign(d.missing_cells)} tone="orange" />
        <Stat label="Duplicate rows" value={`${int(preview.before.duplicate_rows)} → ${int(preview.after.duplicate_rows)}`} hint={sign(d.duplicate_rows)} tone="green" />
      </StatGrid>
      {preview.warnings.map((w) => <div key={w} className="alert alert-warn">{w}</div>)}
      <DataTable columns={['Operation', 'Rows affected', 'Result']} rows={preview.operations.map((o) => [o.type.replace(/_/g, ' '), int(o.rows_affected), <span key="m" className="wrap">{o.message}</span>])} />
      {preview.dtype_changes.length > 0 && (
        <DataTable columns={['Column', 'Type before', 'Type after']} rows={preview.dtype_changes.map((c) => [c.column, c.from, c.to])} />
      )}
      {preview.numeric_changes.length > 0 && (
        <DataTable
          columns={['Column', 'Mean before', 'Mean after', 'Min → Max before', 'Min → Max after']}
          rows={preview.numeric_changes.map((c) => [c.column, num(c.before.mean), num(c.after.mean), `${num(c.before.min)} → ${num(c.before.max)}`, `${num(c.after.min)} → ${num(c.after.max)}`])}
        />
      )}
      <div className="two-col">
        <div><h4>Sample before</h4><DataTable columns={cols(preview.sample_before)} rows={preview.sample_before.slice(0, 5).map((r) => cols(preview.sample_before).map((c) => cell(r[c])))} /></div>
        <div><h4>Sample after</h4><DataTable columns={cols(preview.sample_after)} rows={preview.sample_after.slice(0, 5).map((r) => cols(preview.sample_after).map((c) => cell(r[c])))} /></div>
      </div>
    </div>
  )
}

function ManualBuilder({ columns, onAdd }: { columns: string[]; onAdd: (label: string, op: Op) => void }) {
  const [type, setType] = useState<(typeof OP_TYPES)[number]>('fill_missing')
  const [column, setColumn] = useState('')
  const [strategy, setStrategy] = useState('median')
  const [value, setValue] = useState('')
  const [dtype, setDtype] = useState('numeric')
  const [method, setMethod] = useState('iqr')
  const [action, setAction] = useState('clip')
  const [lower, setLower] = useState(false)
  const [min, setMin] = useState('')
  const [max, setMax] = useState('')
  const [error, setError] = useState('')

  const col = column && columns.includes(column) ? column : columns[0] ?? ''

  const add = () => {
    setError('')
    let op: Op
    switch (type) {
      case 'fill_missing': op = { type, column: col, strategy, ...(strategy === 'constant' ? { value } : {}) }; break
      case 'drop_columns': op = { type, columns: [col] }; break
      case 'convert_dtype': op = { type, column: col, dtype }; break
      case 'handle_outliers': op = { type, column: col, method: method === 'zscore' ? 'zscore' : 'iqr', action }; break
      case 'standardize_text': op = { type, column: col, strip: true, lowercase: lower, collapse_whitespace: true }; break
      case 'replace_invalid': op = { type, column: col }; break
      case 'scale_column': op = { type, column: col, method: method === 'standard' ? 'standard' : 'minmax' }; break
      case 'invalid_to_missing': {
        const lo = min === '' ? undefined : Number(min)
        const hi = max === '' ? undefined : Number(max)
        if (lo === undefined && hi === undefined) return setError('Enter a minimum and/or maximum.')
        op = { type, column: col, ...(lo !== undefined ? { minimum: lo } : {}), ...(hi !== undefined ? { maximum: hi } : {}) }
        break
      }
      default: op = { type }
    }
    if (type === 'fill_missing' && strategy === 'constant' && value === '') return setError('Enter the constant value.')
    onAdd('Manual', op)
  }

  const needsColumn = !NO_COLUMN.has(type)
  return (
    <Panel title="Manual operation" subtitle="Build a custom cleaning step and add it to the queue.">
      <div className="form-grid">
        <Field label="Operation">
          <select value={type} onChange={(e) => setType(e.target.value as typeof type)}>
            {OP_TYPES.map((t) => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
          </select>
        </Field>
        {needsColumn && (
          <Field label="Column"><select value={col} onChange={(e) => setColumn(e.target.value)}>{columns.map((c) => <option key={c}>{c}</option>)}</select></Field>
        )}
        {type === 'fill_missing' && <Field label="Strategy"><select value={strategy} onChange={(e) => setStrategy(e.target.value)}>{['mean', 'median', 'mode', 'constant'].map((s) => <option key={s}>{s}</option>)}</select></Field>}
        {type === 'fill_missing' && strategy === 'constant' && <Field label="Value"><input value={value} onChange={(e) => setValue(e.target.value)} /></Field>}
        {type === 'convert_dtype' && <Field label="Convert to"><select value={dtype} onChange={(e) => setDtype(e.target.value)}>{['numeric', 'datetime', 'string', 'category', 'boolean'].map((s) => <option key={s}>{s}</option>)}</select></Field>}
        {type === 'handle_outliers' && <>
          <Field label="Method"><select value={method} onChange={(e) => setMethod(e.target.value)}><option value="iqr">IQR (1.5×)</option><option value="zscore">Z-score (3σ)</option></select></Field>
          <Field label="Action"><select value={action} onChange={(e) => setAction(e.target.value)}><option value="clip">Clip to bounds</option><option value="set_missing">Set to missing</option><option value="remove">Remove rows</option></select></Field>
        </>}
        {type === 'scale_column' && <Field label="Scaling"><select value={method} onChange={(e) => setMethod(e.target.value)}><option value="minmax">Min–max (0–1)</option><option value="standard">Standardize (z-score)</option></select></Field>}
        {type === 'standardize_text' && <Field label="Lower-case"><input type="checkbox" checked={lower} onChange={(e) => setLower(e.target.checked)} /></Field>}
        {type === 'invalid_to_missing' && <>
          <Field label="Minimum allowed"><input type="number" value={min} onChange={(e) => setMin(e.target.value)} /></Field>
          <Field label="Maximum allowed"><input type="number" value={max} onChange={(e) => setMax(e.target.value)} /></Field>
        </>}
        <div className="form-action"><button className="ghost-button" onClick={add} disabled={needsColumn && !col}><Plus size={14} /> Add to queue</button></div>
      </div>
      <ErrorNote message={error} />
    </Panel>
  )
}
