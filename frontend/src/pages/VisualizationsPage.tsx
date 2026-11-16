import { useState } from 'react'
import { api } from '../lib/api'
import type { Metadata, PageCtx } from '../types'
import { AiExplain } from '../components/AiExplain'
import {
  BoxPlot, CategoryBars, Heatmap, HistogramChart, LineSeries, ScatterView,
  type BoxStat, type Bin,
} from '../components/charts'
import { Busy, Empty, ErrorNote, Field, Panel, Stat, StatGrid } from '../components/ui'
import { useAsync } from '../hooks'
import { num } from '../lib/format'

const TYPES = [
  { id: 'histogram', label: 'Histogram', x: 'numeric' },
  { id: 'distribution', label: 'Distribution', x: 'numeric' },
  { id: 'bar', label: 'Bar chart', x: 'any', y: 'numeric?', agg: true },
  { id: 'line', label: 'Line chart', x: 'any', y: 'numeric', agg: true },
  { id: 'scatter', label: 'Scatter plot', x: 'numeric', y: 'numeric', group: true },
  { id: 'box', label: 'Box plot', x: 'numeric', group: true },
  { id: 'heatmap', label: 'Correlation heatmap' },
  { id: 'missing', label: 'Missing values' },
] as const

type ChartResponse = {
  chart_type: string
  bins?: Bin[]
  n?: number
  summary?: Record<string, number | null>
  data?: { category?: string; value?: number; x?: string | number; y?: number; column?: string; percentage?: number; missing?: number }[]
  series?: { name: string; data: { x: number; y: number }[] }[]
  boxes?: BoxStat[]
  columns?: string[]
  matrix?: (number | null)[][]
  correlation?: number | null
  total_points?: number
  displayed_points?: number
  aggregation?: string
}

export default function VisualizationsPage({ datasetId }: PageCtx) {
  const meta = useAsync(async () => (await api.get<Metadata>(`/datasets/${datasetId}/metadata`)).data, datasetId)
  const [type, setType] = useState<(typeof TYPES)[number]['id']>('histogram')
  const [x, setX] = useState('')
  const [y, setY] = useState('')
  const [group, setGroup] = useState('')
  const [agg, setAgg] = useState('mean')
  const [bins, setBins] = useState(12)

  const columns = meta.data?.columns ?? []
  const isNumeric = (dtype: string) => /int|float/.test(dtype)
  const numeric = columns.filter((c) => isNumeric(c.dtype))
  const spec = TYPES.find((t) => t.id === type)!
  const xOptions = ('x' in spec && spec.x === 'numeric') ? numeric : columns
  const yOptions = numeric

  const xValue = xOptions.some((c) => c.name === x) ? x : xOptions[0]?.name ?? ''
  const yValue = yOptions.some((c) => c.name === y) ? y : (type === 'bar' ? '' : yOptions.find((c) => c.name !== xValue)?.name ?? yOptions[0]?.name ?? '')
  const needsX = 'x' in spec
  const needsY = 'y' in spec
  const ready = !!meta.data && (!needsX || !!xValue) && (!needsY || type === 'bar' || !!yValue)

  const params: Record<string, string | number> = { chart_type: type, bins }
  if (needsX && xValue) params.x_column = xValue
  if (needsY && yValue) params.y_column = yValue
  if ('group' in spec && group) params.group_by = group
  if ('agg' in spec) params.agg = type === 'bar' && !yValue ? 'count' : agg
  const key = ready ? `${datasetId}|${JSON.stringify(params)}` : null

  const chart = useAsync(
    async () => (await api.get<ChartResponse>(`/datasets/${datasetId}/charts`, { params })).data,
    key,
  )

  if (meta.loading) return <Busy />
  if (meta.error) return <ErrorNote message={meta.error} onRetry={meta.reload} />
  if (columns.length === 0) return <Empty title="No columns found" />

  const c = chart.data
  return (
    <>
      <Panel title="Chart builder" subtitle="Charts are computed from the actual dataset values.">
        <div className="form-grid">
          <Field label="Chart type">
            <select value={type} onChange={(e) => setType(e.target.value as typeof type)}>
              {TYPES.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </Field>
          {needsX && (
            <Field label={type === 'scatter' ? 'X axis' : 'Column'}>
              <select value={xValue} onChange={(e) => setX(e.target.value)}>
                {xOptions.map((o) => <option key={o.name}>{o.name}</option>)}
              </select>
            </Field>
          )}
          {needsY && (
            <Field label={type === 'bar' ? 'Value column (optional)' : 'Y axis'}>
              <select value={yValue} onChange={(e) => setY(e.target.value)}>
                {type === 'bar' && <option value="">(count rows)</option>}
                {yOptions.map((o) => <option key={o.name}>{o.name}</option>)}
              </select>
            </Field>
          )}
          {'agg' in spec && (type !== 'bar' || yValue) && (
            <Field label="Aggregation">
              <select value={agg} onChange={(e) => setAgg(e.target.value)}>
                {['mean', 'sum', 'median', 'min', 'max'].map((a) => <option key={a}>{a}</option>)}
              </select>
            </Field>
          )}
          {'group' in spec && (
            <Field label="Group / colour by">
              <select value={group} onChange={(e) => setGroup(e.target.value)}>
                <option value="">(none)</option>
                {columns.filter((o) => !isNumeric(o.dtype) || o.unique <= 12).map((o) => <option key={o.name}>{o.name}</option>)}
              </select>
            </Field>
          )}
          {(type === 'histogram' || type === 'distribution') && (
            <Field label={`Bins: ${bins}`}>
              <input type="range" min={3} max={40} value={bins} onChange={(e) => setBins(Number(e.target.value))} />
            </Field>
          )}
        </div>
      </Panel>

      <Panel title={spec.label} subtitle={ready ? undefined : 'Pick the required columns.'}>
        {chart.loading && <Busy text="Computing chart…" />}
        <ErrorNote message={chart.error} onRetry={chart.reload} />
        {!chart.loading && c && (
          <>
            {(c.chart_type === 'histogram' || c.chart_type === 'distribution') && c.bins && (
              <>
                <HistogramChart bins={c.bins} label={`Histogram of ${xValue}`} />
                {c.summary && (
                  <StatGrid>
                    <Stat label="Mean" value={num(c.summary.mean)} />
                    <Stat label="Median" value={num(c.summary.median)} />
                    <Stat label="Std dev" value={num(c.summary.std)} />
                    <Stat label="Skewness" value={num(c.summary.skewness)} />
                  </StatGrid>
                )}
              </>
            )}
            {c.chart_type === 'bar' && c.data && (
              <CategoryBars
                valueLabel={c.aggregation ?? 'count'}
                data={c.data.map((d) => ({ name: String(d.category), value: d.value ?? 0 }))}
              />
            )}
            {c.chart_type === 'line' && c.data && (
              <LineSeries xLabel={xValue} yLabel={`${c.aggregation} of ${yValue}`} data={c.data.map((d) => ({ x: d.x ?? '', y: d.y ?? 0 }))} />
            )}
            {c.chart_type === 'scatter' && (
              <>
                <ScatterView
                  xLabel={xValue}
                  yLabel={yValue}
                  series={c.series ?? [{ name: 'points', data: (c.data ?? []).map((d) => ({ x: Number(d.x), y: Number(d.y) })) }]}
                />
                <p className="muted small">
                  {c.displayed_points} of {c.total_points} points shown
                  {c.correlation !== null && c.correlation !== undefined && ` · Pearson r = ${c.correlation.toFixed(3)}`}
                </p>
              </>
            )}
            {c.chart_type === 'box' && c.boxes && <BoxPlot boxes={c.boxes} />}
            {c.chart_type === 'heatmap' && c.columns && c.matrix && <Heatmap columns={c.columns} matrix={c.matrix} />}
            {c.chart_type === 'missing' && c.data && (
              <CategoryBars
                horizontal
                color="#f59e0b"
                valueLabel="% missing"
                data={c.data.filter((d) => (d.missing ?? 0) > 0).map((d) => ({ name: String(d.column), value: d.percentage ?? 0 }))}
              />
            )}
            {c.chart_type === 'missing' && c.data && c.data.every((d) => !d.missing) && (
              <Empty title="No missing values" text="Nothing to chart: every column is complete." />
            )}
            <div className="spacer" />
            <AiExplain datasetId={datasetId} kind="chart" label="Explain this chart" chart={{
              chart_type: type, x_column: params.x_column, y_column: params.y_column,
              group_by: params.group_by, agg: params.agg ?? 'count', bins,
            }} />
          </>
        )}
      </Panel>
    </>
  )
}

