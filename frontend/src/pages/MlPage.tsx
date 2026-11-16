import { useState } from 'react'
import { BrainCircuit, GitCompare, Play, RefreshCcw, Trash2 } from 'lucide-react'
import { api, apiError } from '../lib/api'
import { int, num, when } from '../lib/format'
import type { FeatureSchema, Metadata, MlRun, PageCtx } from '../types'
import { AiExplain } from '../components/AiExplain'
import { CategoryBars, ScatterView } from '../components/charts'
import { Badge, Busy, DataTable, Empty, ErrorNote, Field, Panel, Stat, StatGrid, Tabs } from '../components/ui'
import { useAsync } from '../hooks'

type Task = 'regression' | 'classification' | 'clustering'
type Catalog = { algorithms: Record<Task, string[]> }
const HP: Record<string, { key: string; label: string; min: number; max: number; step: number }[]> = {
  random_forest: [{ key: 'n_estimators', label: 'Trees', min: 5, max: 500, step: 5 }, { key: 'max_depth', label: 'Max depth', min: 1, max: 50, step: 1 }],
  gradient_boosting: [{ key: 'n_estimators', label: 'Stages', min: 5, max: 500, step: 5 }, { key: 'learning_rate', label: 'Learning rate', min: 0.001, max: 1, step: 0.01 }, { key: 'max_depth', label: 'Max depth', min: 1, max: 20, step: 1 }],
  ridge: [{ key: 'alpha', label: 'Alpha', min: 0.0001, max: 1000, step: 0.1 }],
  logistic_regression: [{ key: 'C', label: 'C (inverse regularisation)', min: 0.0001, max: 1000, step: 0.1 }],
}

export default function MlPage({ datasetId, notify, fail }: PageCtx) {
  const meta = useAsync(async () => (await api.get<Metadata>(`/datasets/${datasetId}/metadata`)).data, datasetId)
  const catalog = useAsync(async () => (await api.get<Catalog>('/ml/catalog')).data, 'catalog')
  const runs = useAsync(async () => (await api.get<MlRun[]>(`/datasets/${datasetId}/ml/runs`)).data, datasetId)

  const [tab, setTab] = useState<'train' | 'models'>('train')
  const [task, setTask] = useState<Task>('regression')
  const [target, setTarget] = useState('')
  const [algorithm, setAlgorithm] = useState('')
  const [excluded, setExcluded] = useState<string[]>([])
  const [testSize, setTestSize] = useState(0.2)
  const [cv, setCv] = useState(5)
  const [fsK, setFsK] = useState(0)
  const [clusters, setClusters] = useState(0)
  const [hp, setHp] = useState<Record<string, string>>({})
  const [name, setName] = useState('')
  const [seed, setSeed] = useState('')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const [compare, setCompare] = useState<MlRun[]>([])

  const columns = meta.data?.columns ?? []
  const algos = catalog.data?.algorithms[task] ?? []
  const algo = algos.includes(algorithm) ? algorithm : algos[0] ?? ''
  const targetOptions = columns.filter((c) => (task === 'regression' ? /int|float/.test(c.dtype) : true))
  const tgt = task === 'clustering' ? '' : targetOptions.some((c) => c.name === target) ? target : targetOptions[0]?.name ?? ''
  const featureOptions = columns.filter((c) => c.name !== tgt)

  const body = () => {
    const hyper: Record<string, number> = {}
    for (const h of HP[algo] ?? []) if (hp[h.key] !== undefined && hp[h.key] !== '') hyper[h.key] = Number(hp[h.key])
    return {
      task, target: task === 'clustering' ? null : tgt,
      features: excluded.length ? featureOptions.map((c) => c.name).filter((n) => !excluded.includes(n)) : null,
      test_size: testSize, cv_folds: cv,
      feature_selection: fsK > 0 ? { method: 'k_best', k: fsK } : { method: 'none' },
      n_clusters: task === 'clustering' && clusters >= 2 ? clusters : null,
      hyperparameters: hyper, model_name: name || null, random_state: seed === '' ? null : Number(seed),
    }
  }

  const train = async () => {
    setBusy('train'); setError('')
    try {
      const r = await api.post<MlRun>(`/datasets/${datasetId}/ml/train`, { ...body(), algorithm: algo })
      notify(`Trained ${r.data.model_name} v${r.data.version}.`)
      runs.reload(); setSelected(r.data.id); setTab('models')
    } catch (err) { setError(apiError(err)) } finally { setBusy('') }
  }

  const compareAll = async () => {
    setBusy('compare'); setError('')
    try {
      const r = await api.post<{ runs: MlRun[]; errors: string[] }>(`/datasets/${datasetId}/ml/compare`, { ...body(), model_name: null, algorithms: algos })
      setCompare(r.data.runs)
      notify(`Compared ${r.data.runs.length} algorithms${r.data.errors.length ? ` (${r.data.errors.length} failed)` : ''}.`)
      runs.reload(); setTab('models')
    } catch (err) { setError(apiError(err)) } finally { setBusy('') }
  }

  if (meta.loading || catalog.loading) return <Busy />
  if (meta.error) return <ErrorNote message={meta.error} onRetry={meta.reload} />

  const allRuns = runs.data ?? []
  const current = allRuns.find((r) => r.id === selected) ?? null

  return (
    <>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'train', label: 'Train a model' }, { id: 'models', label: `Trained models (${allRuns.length})` }]} />

      {tab === 'train' && (
        <Panel title="Machine learning studio" subtitle="Real scikit-learn training with a held-out test set, preprocessing fitted on training data only, and a naive baseline for comparison.">
          <div className="form-grid">
            <Field label="Task">
              <select value={task} onChange={(e) => { setTask(e.target.value as Task); setAlgorithm(''); setTarget(''); setHp({}) }}>
                <option value="regression">Regression (predict a number)</option>
                <option value="classification">Classification (predict a category)</option>
                <option value="clustering">Clustering (find groups)</option>
              </select>
            </Field>
            {task !== 'clustering' && (
              <Field label="Target column" hint="The column the model learns to predict.">
                <select value={tgt} onChange={(e) => setTarget(e.target.value)}>{targetOptions.map((c) => <option key={c.name}>{c.name}</option>)}</select>
              </Field>
            )}
            <Field label="Algorithm"><select value={algo} onChange={(e) => { setAlgorithm(e.target.value); setHp({}) }}>{algos.map((a) => <option key={a} value={a}>{a.replace(/_/g, ' ')}</option>)}</select></Field>
            {task !== 'clustering' && <Field label={`Test size: ${Math.round(testSize * 100)}%`}><input type="range" min={0.1} max={0.5} step={0.05} value={testSize} onChange={(e) => setTestSize(Number(e.target.value))} /></Field>}
            {task !== 'clustering' && <Field label="Cross-validation folds" hint="0 disables"><input type="number" min={0} max={10} value={cv} onChange={(e) => setCv(Number(e.target.value))} /></Field>}
            {task !== 'clustering' && <Field label="Keep best K features (0 = all)" hint="Selection runs inside the training pipeline."><input type="number" min={0} max={200} value={fsK} onChange={(e) => setFsK(Number(e.target.value))} /></Field>}
            {task === 'clustering' && <Field label="Number of clusters (0 = choose automatically)" hint="Auto picks the best silhouette score for k = 2…8."><input type="number" min={0} max={20} value={clusters} onChange={(e) => setClusters(Number(e.target.value))} /></Field>}
            {(HP[algo] ?? []).map((h) => (
              <Field key={h.key} label={h.label} hint="Leave empty for default">
                <input type="number" min={h.min} max={h.max} step={h.step} value={hp[h.key] ?? ''} onChange={(e) => setHp({ ...hp, [h.key]: e.target.value })} />
              </Field>
            ))}
            <Field label="Model name (optional)" hint="Same name → new version"><input value={name} maxLength={100} onChange={(e) => setName(e.target.value)} /></Field>
            <Field label="Random seed (optional)" hint="Same seed → same result"><input type="number" min={0} value={seed} onChange={(e) => setSeed(e.target.value)} /></Field>
          </div>

          <details className="features-box">
            <summary>Features: {featureOptions.length - excluded.length} of {featureOptions.length} selected</summary>
            <p className="muted small">Identifier-like, constant and target-duplicate columns are excluded automatically and reported after training.</p>
            <div className="check-grid">
              {featureOptions.map((c) => (
                <label key={c.name}><input type="checkbox" checked={!excluded.includes(c.name)} onChange={(e) => setExcluded(e.target.checked ? excluded.filter((n) => n !== c.name) : [...excluded, c.name])} /> {c.name} <small className="muted">{c.dtype}</small></label>
              ))}
            </div>
          </details>

          <ErrorNote message={error} />
          <div className="button-row">
            <button className="primary-button" onClick={() => void train()} disabled={!!busy || !algo || (task !== 'clustering' && !tgt)}><Play size={14} /> {busy === 'train' ? 'Training…' : 'Train model'}</button>
            <button className="ghost-button" onClick={() => void compareAll()} disabled={!!busy || algos.length < 2 || (task !== 'clustering' && !tgt)}><GitCompare size={14} /> {busy === 'compare' ? 'Comparing…' : `Compare all ${algos.length} algorithms`}</button>
          </div>
          {busy && <Busy text="Training with scikit-learn…" />}
        </Panel>
      )}

      {tab === 'models' && (
        <>
          {compare.length > 0 && (
            <Panel title="Algorithm comparison" subtitle="Same data and split; ranked by the primary metric." actions={<button className="ghost-button" onClick={() => setCompare([])}>Dismiss</button>}>
              <DataTable
                columns={['Rank', 'Model', 'Primary metric', 'Value', 'Baseline', 'Time']}
                rows={compare.map((r, i) => [i + 1, <button key="n" className="link-button" onClick={() => setSelected(r.id)}>{r.algorithm.replace(/_/g, ' ')}</button>, r.primary_metric, num(r.primary_metric_value, 3), baselineText(r), `${r.duration_ms} ms`])}
              />
            </Panel>
          )}
          <Panel title="Model registry" subtitle="Every run is stored with its configuration, metrics, dataset fingerprint and artifact.">
            {runs.loading ? <Busy /> : allRuns.length === 0 ? (
              <Empty icon={<BrainCircuit size={24} />} title="No models yet" text="Train your first model on the Train tab." />
            ) : (
              <DataTable
                columns={['Model', 'Ver.', 'Task', 'Target', 'Primary metric', 'Trained', '']}
                rows={allRuns.map((r) => [
                  <button key="n" className="link-button" onClick={() => setSelected(r.id)}>{r.model_name}</button>,
                  `v${r.version}`, <Badge key="t" tone="purple">{r.task}</Badge>, r.target ?? '—',
                  `${r.primary_metric}: ${num(r.primary_metric_value, 3)}`, when(r.created_at),
                  <button key="d" className="icon-button small" aria-label="Delete run" onClick={async () => {
                    if (!window.confirm(`Delete ${r.model_name} v${r.version} and its stored model file?`)) return
                    try { await api.delete(`/ml/runs/${r.id}`); if (selected === r.id) setSelected(null); runs.reload() } catch (e) { fail(apiError(e)) }
                  }}><Trash2 size={14} /></button>,
                ])}
              />
            )}
          </Panel>
          {current && <RunDetail key={current.id} runId={current.id} datasetId={datasetId} notify={notify} fail={fail} onChanged={runs.reload} />}
        </>
      )}
    </>
  )
}

function baselineText(r: MlRun): string {
  if (r.task === 'regression') return `R² ${num(r.metrics.baseline_r2, 3)}`
  if (r.task === 'classification') return `acc ${num(r.metrics.baseline_accuracy, 3)}`
  return '—'
}

function RunDetail({ runId, datasetId, notify, fail, onChanged }: { runId: number; datasetId: number; notify: (m: string) => void; fail: (m: string) => void; onChanged: () => void }) {
  const run = useAsync(async () => (await api.get<MlRun>(`/ml/runs/${runId}`)).data, runId)
  const [repro, setRepro] = useState<string>('')
  if (run.loading) return <Busy />
  if (run.error) return <ErrorNote message={run.error} />
  const r = run.data
  if (!r) return null
  const d = r.details ?? {}
  const m = r.metrics

  const reproduce = async () => {
    setRepro('Retraining…')
    try {
      const res = await api.post<{ metrics_identical: boolean; new_run: MlRun }>(`/ml/runs/${r.id}/reproduce`)
      setRepro(res.data.metrics_identical ? `Reproduced exactly: all metrics identical (saved as v${res.data.new_run.version}).` : 'Reproduced, but some metrics differ.')
      notify('Run reproduced.'); onChanged()
    } catch (err) { setRepro(''); fail(apiError(err)) }
  }

  return (
    <Panel
      title={`${r.model_name} · v${r.version}`}
      subtitle={`${r.task} · ${r.algorithm.replace(/_/g, ' ')} · trained ${when(r.created_at)} in ${r.duration_ms} ms`}
      actions={<button className="ghost-button" onClick={() => void reproduce()}><RefreshCcw size={14} /> Reproduce</button>}
    >
      {repro && <div className="alert alert-success">{repro}</div>}
      {(d.warnings ?? []).map((w) => <div key={w} className="alert alert-warn">{w}</div>)}

      <StatGrid>
        {r.task === 'regression' && <>
          <Stat label="R² (test)" value={num(m.r2, 3)} hint={`baseline ${num(m.baseline_r2, 3)}`} tone="purple" />
          <Stat label="MAE" value={num(m.mae)} hint={`baseline ${num(m.baseline_mae)}`} tone="blue" />
          <Stat label="RMSE" value={num(m.rmse)} tone="orange" />
          {m.cv_r2_mean !== undefined && <Stat label="CV R² (train)" value={num(m.cv_r2_mean, 3)} hint={`± ${num(m.cv_r2_std, 2)}`} tone="green" />}
        </>}
        {r.task === 'classification' && <>
          <Stat label="Accuracy" value={num(m.accuracy, 3)} hint={`baseline ${num(m.baseline_accuracy, 3)}`} tone="purple" />
          <Stat label="F1 (weighted)" value={num(m.f1_weighted, 3)} tone="blue" />
          <Stat label="Balanced accuracy" value={num(m.balanced_accuracy, 3)} tone="orange" />
          {m.roc_auc !== undefined && <Stat label="ROC AUC" value={num(m.roc_auc, 3)} tone="green" />}
        </>}
        {r.task === 'clustering' && <>
          <Stat label="Clusters" value={int(m.n_clusters)} tone="purple" />
          <Stat label="Silhouette" value={num(m.silhouette, 3)} hint="−1 … 1, higher is better" tone="blue" />
          <Stat label="Davies–Bouldin" value={num(m.davies_bouldin, 3)} hint="lower is better" tone="orange" />
        </>}
      </StatGrid>
      {d.split && <p className="muted small">Trained on {int(d.split.train_rows)} rows, evaluated on {int(d.split.test_rows)} held-out rows. Dataset v{r.dataset_version_number}, SHA-256 {r.dataset_sha256?.slice(0, 12)}…, scikit-learn {r.environment?.scikit_learn}.</p>}
      {d.note && <p className="muted small">{d.note}</p>}

      <div className="two-col">
        {(d.feature_importance ?? []).length > 0 && <div><h4>Feature importance</h4><CategoryBars horizontal valueLabel="share" data={d.feature_importance!.slice(0, 10).map((f) => ({ name: f.feature, value: f.importance }))} /></div>}
        {(d.predictions_sample ?? []).length > 0 && <div><h4>Predicted vs actual (test set)</h4><ScatterView xLabel="actual" yLabel="predicted" series={[{ name: 'test rows', data: d.predictions_sample!.map((p) => ({ x: p.actual, y: p.predicted })) }]} /></div>}
        {d.confusion_matrix && d.classes && <div><h4>Confusion matrix (test set)</h4><DataTable columns={['actual ↓ / predicted →', ...d.classes]} rows={d.confusion_matrix.map((row, i) => [<strong key="c">{d.classes![i]}</strong>, ...row])} /></div>}
        {d.pca_projection && <div><h4>Clusters (2-D PCA projection)</h4><ScatterView xLabel="PC1" yLabel="PC2" series={[...new Set(d.pca_projection.map((p) => p.cluster))].sort().map((c) => ({ name: `Cluster ${c}`, data: d.pca_projection!.filter((p) => p.cluster === c) }))} /></div>}
      </div>

      {d.cluster_profiles && <DataTable columns={['Cluster', 'Size', '% of rows', 'Mean values (first features)']} rows={d.cluster_profiles.map((c) => [c.cluster, int(c.size), `${c.percentage}%`, Object.entries(c.numeric_means).slice(0, 4).map(([k, v]) => `${k}: ${num(v)}`).join(' · ')])} />}
      {(d.excluded_features ?? []).length > 0 && <details><summary>{d.excluded_features!.length} column(s) excluded automatically</summary><DataTable columns={['Column', 'Reason']} rows={d.excluded_features!.map((e) => [e.column, e.reason])} /></details>}
      <details><summary>All metrics &amp; configuration</summary>
        <DataTable columns={['Metric', 'Value']} rows={Object.entries(m).map(([k, v]) => [k, num(v, 4)])} />
        <pre className="json-view">{JSON.stringify(r.config, null, 2)}</pre>
      </details>

      <div className="spacer" />
      <AiExplain datasetId={datasetId} kind="ml_run" runId={r.id} label="Explain these results" />
      <PredictForm runId={r.id} />
    </Panel>
  )
}

function PredictForm({ runId }: { runId: number }) {
  const schema = useAsync(async () => (await api.get<{ features: FeatureSchema[] }>(`/ml/runs/${runId}/schema`)).data.features, runId)
  const [vals, setVals] = useState<Record<string, string>>({})
  const [out, setOut] = useState<{ predictions: { prediction?: string | number; cluster?: number; probabilities?: Record<string, number> }[]; warnings: string[] } | null>(null)
  const [error, setError] = useState('')
  if (!schema.data) return null

  const go = async () => {
    setError('')
    const rec: Record<string, string | number> = {}
    for (const f of schema.data!) {
      const v = vals[f.name]
      if (v === undefined || v === '') continue
      rec[f.name] = f.kind === 'numeric' ? Number(v) : v
    }
    try { setOut((await api.post(`/ml/runs/${runId}/predict`, { records: [rec] })).data) } catch (e) { setOut(null); setError(apiError(e)) }
  }
  const p = out?.predictions[0]
  return (
    <div className="predict-box">
      <h4>Try the model</h4>
      <p className="muted small">Leave a field empty to use the training-set default (median / most frequent).</p>
      <div className="form-grid">
        {schema.data.map((f) => (
          <Field key={f.name} label={f.name} hint={f.kind === 'numeric' ? `typical ${num(f.median)} (${num(f.min)}–${num(f.max)})` : undefined}>
            {f.kind === 'numeric'
              ? <input type="number" value={vals[f.name] ?? ''} onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })} />
              : <select value={vals[f.name] ?? ''} onChange={(e) => setVals({ ...vals, [f.name]: e.target.value })}><option value="">(default)</option>{(f.values ?? []).map((v) => <option key={v}>{v}</option>)}</select>}
          </Field>
        ))}
      </div>
      <button className="primary-button" onClick={() => void go()}>Predict</button>
      <ErrorNote message={error} />
      {p && (
        <div className="prediction">
          <strong>{p.cluster !== undefined ? `Cluster ${p.cluster}` : typeof p.prediction === 'number' ? num(p.prediction, 5) : String(p.prediction)}</strong>
          {p.probabilities && <div>{Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, v]) => <Badge key={k} tone="purple">{k}: {(v * 100).toFixed(1)}%</Badge>)}</div>}
          {out!.warnings.map((w) => <small key={w} className="muted">{w}</small>)}
        </div>
      )}
    </div>
  )
}

