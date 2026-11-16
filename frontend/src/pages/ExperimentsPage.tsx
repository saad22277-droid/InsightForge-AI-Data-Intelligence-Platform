import { useState } from 'react'
import { FlaskConical, GitCompare } from 'lucide-react'
import { api, apiError } from '../lib/api'
import { int, num, when } from '../lib/format'
import type { MlRun } from '../types'
import { Badge, Busy, DataTable, Empty, ErrorNote, Panel, Stat, StatGrid } from '../components/ui'
import { useAsync } from '../hooks'

type Summary = { total_runs: number; by_task: Record<string, number>; best_models: { run_id: number; dataset_id: number; task: string; target: string | null; model_name: string; version: number; primary_metric: string; value: number }[] }
type Compare = { task: string; runs: MlRun[]; metrics: string[]; best_by_metric: Record<string, number> }

export default function ExperimentsPage() {
  const [task, setTask] = useState('')
  const exps = useAsync(async () => (await api.get<MlRun[]>('/experiments', { params: task ? { task } : {} })).data, `exp:${task}`)
  const summary = useAsync(async () => (await api.get<Summary>('/experiments/summary')).data, `sum:${exps.data?.length ?? 0}`)
  const [picked, setPicked] = useState<number[]>([])
  const [cmp, setCmp] = useState<Compare | null>(null)
  const [error, setError] = useState('')

  const toggle = (id: number) => setPicked((p) => (p.includes(id) ? p.filter((i) => i !== id) : [...p, id].slice(-6)))
  const compare = async () => {
    setError('')
    try {
      setCmp((await api.get<Compare>('/experiments/compare', { params: { ids: picked.join(',') } })).data)
    } catch (err) { setCmp(null); setError(apiError(err)) }
  }

  const rows = exps.data ?? []
  return (
    <>
      <StatGrid>
        <Stat label="Tracked runs" value={int(summary.data?.total_runs)} tone="purple" />
        <Stat label="Regression" value={int(summary.data?.by_task.regression)} tone="blue" />
        <Stat label="Classification" value={int(summary.data?.by_task.classification)} tone="green" />
        <Stat label="Clustering" value={int(summary.data?.by_task.clustering)} tone="orange" />
      </StatGrid>

      {(summary.data?.best_models ?? []).length > 0 && (
        <Panel title="Best models" subtitle="Highest primary metric per dataset, task and target">
          <DataTable columns={['Model', 'Task', 'Target', 'Metric', 'Value']} rows={summary.data!.best_models.map((b) => [`${b.model_name} v${b.version}`, <Badge key="t" tone="purple">{b.task}</Badge>, b.target ?? '—', b.primary_metric, num(b.value, 3)])} />
        </Panel>
      )}

      <Panel
        title="Experiment tracker"
        subtitle="Every training run with its configuration, metrics and dataset fingerprint. Select 2–6 runs of the same task to compare."
        actions={
          <div className="button-row">
            <select value={task} onChange={(e) => { setTask(e.target.value); setPicked([]); setCmp(null) }} aria-label="Filter by task">
              <option value="">All tasks</option><option value="regression">Regression</option><option value="classification">Classification</option><option value="clustering">Clustering</option>
            </select>
            <button className="primary-button" disabled={picked.length < 2} onClick={() => void compare()}><GitCompare size={14} /> Compare ({picked.length})</button>
          </div>
        }
      >
        {exps.loading && <Busy />}
        <ErrorNote message={exps.error || error} onRetry={exps.reload} />
        {!exps.loading && rows.length === 0 && <Empty icon={<FlaskConical size={24} />} title="No experiments yet" text="Train a model in Machine Learning and it will appear here." />}
        {rows.length > 0 && (
          <DataTable
            columns={['', 'Run', 'Dataset', 'Model', 'Task', 'Target', 'Primary metric', 'Test size', 'Trained']}
            rows={rows.map((r) => [
              <input key="c" type="checkbox" aria-label={`Select run ${r.id}`} checked={picked.includes(r.id)} onChange={() => toggle(r.id)} />,
              `#${r.id}`, r.dataset_filename ?? r.dataset_id, `${r.model_name} v${r.version}`,
              <Badge key="t" tone="purple">{r.task}</Badge>, r.target ?? '—',
              `${r.primary_metric}: ${num(r.primary_metric_value, 3)}`,
              String((r.config as { test_size?: number }).test_size ?? '—'),
              when(r.created_at),
            ])}
          />
        )}
      </Panel>

      {cmp && (
        <Panel title="Model comparison" subtitle={`${cmp.task} · best value per metric highlighted`}>
          <DataTable
            columns={['Metric', ...cmp.runs.map((r) => `#${r.id} ${r.algorithm.replace(/_/g, ' ')}`)]}
            rows={[
              ['Dataset version', ...cmp.runs.map((r) => `v${r.dataset_version_number}`)],
              ['Random seed', ...cmp.runs.map((r) => String((r.config as { random_state?: number }).random_state ?? '—'))],
              ...cmp.metrics.map((m) => [m, ...cmp.runs.map((r) => (
                cmp.best_by_metric[m] === r.id ? <Badge key={r.id} tone="green">{num(r.metrics[m], 4)}</Badge> : num(r.metrics[m], 4)
              ))]),
            ]}
          />
        </Panel>
      )}
    </>
  )
}
