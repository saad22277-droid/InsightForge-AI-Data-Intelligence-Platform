import { useState } from 'react'
import { Download } from 'lucide-react'
import { api, downloadUrl } from '../lib/api'
import { int, num, pct } from '../lib/format'
import type { Eda, PageCtx } from '../types'
import { AiExplain } from '../components/AiExplain'
import { CategoryBars, Heatmap, HistogramChart } from '../components/charts'
import { Badge, Busy, DataTable, Empty, ErrorNote, Panel, Stat, StatGrid, Tabs } from '../components/ui'
import { useAsync } from '../hooks'

type Tab = 'numeric' | 'categorical' | 'missing' | 'distributions' | 'correlations' | 'outliers' | 'insights'

export default function EdaPage({ datasetId }: PageCtx) {
  const { data: eda, error, loading, reload } = useAsync(
    async () => (await api.get<Eda>(`/datasets/${datasetId}/eda`)).data,
    datasetId,
  )
  const [tab, setTab] = useState<Tab>('numeric')
  const [distColumn, setDistColumn] = useState('')

  if (loading && !eda) return <Busy text="Running exploratory analysis…" />
  if (error) return <ErrorNote message={error} onRetry={reload} />
  if (!eda) return null

  const distColumns = Object.keys(eda.distributions)
  const activeDist = distColumn && eda.distributions[distColumn] ? distColumn : distColumns[0]
  const outlierRows = Object.entries(eda.outliers.columns)
  const q = eda.quality.quality_score

  return (
    <>
      <StatGrid>
        <Stat label="Rows × columns" value={`${int(eda.overview.row_count)} × ${eda.overview.column_count}`} tone="blue" />
        <Stat label="Quality score" value={`${q} / 100`} tone={q >= 80 ? 'green' : q >= 60 ? 'orange' : 'red'} />
        <Stat label="Missing cells" value={int(eda.missing.total_missing_cells)} hint={`${int(eda.missing.rows_with_missing)} rows affected`} tone="orange" />
        <Stat label="Duplicate rows" value={int(eda.duplicates.duplicate_rows)} hint={pct(eda.duplicates.percentage)} tone="purple" />
        <Stat label="Outliers (IQR)" value={int(eda.outliers.total_outliers)} tone="red" />
      </StatGrid>

      <Panel
        title="AI summary"
        subtitle="Gemini explains the computed findings below. Every number is verified against them."
        actions={
          <a className="ghost-button" href={downloadUrl(`/datasets/${datasetId}/exports/columns.csv`)}>
            <Download size={14} /> Column profile CSV
          </a>
        }
      >
        <AiExplain datasetId={datasetId} kind="eda" label="Summarize EDA findings" />
      </Panel>

      <Panel>
        <Tabs
          value={tab}
          onChange={setTab}
          tabs={[
            { id: 'numeric', label: 'Numeric' },
            { id: 'categorical', label: 'Categorical' },
            { id: 'missing', label: 'Missing values' },
            { id: 'distributions', label: 'Distributions' },
            { id: 'correlations', label: 'Correlations' },
            { id: 'outliers', label: 'Outliers' },
            { id: 'insights', label: `Insights (${eda.column_insights.length})` },
          ]}
        />

        {tab === 'numeric' && (
          <DataTable
            empty="This dataset has no numeric columns."
            columns={['Column', 'Count', 'Mean', 'Std', 'Min', 'Median', 'Max', 'Skew', 'Missing']}
            rows={Object.entries(eda.numeric_statistics).map(([c, s]) => [
              <strong key={c}>{c}</strong>, int(s.count), num(s.mean), num(s.std), num(s.min),
              num(s.median), num(s.max), num(s.skewness), int(s.missing),
            ])}
          />
        )}

        {tab === 'categorical' && (
          <DataTable
            empty="This dataset has no categorical columns."
            columns={['Column', 'Distinct', 'Missing', 'Most frequent values']}
            rows={Object.entries(eda.categorical_statistics).map(([c, s]) => [
              <strong key={c}>{c}</strong>, int(s.unique), int(s.missing),
              s.top_values.slice(0, 4).map((t) => `${t.value} (${t.count})`).join(' · '),
            ])}
          />
        )}

        {tab === 'missing' && (
          <>
            {eda.missing.total_missing_cells === 0 ? (
              <Empty title="No missing values" text="Every cell in this dataset has a value." />
            ) : (
              <CategoryBars
                horizontal
                valueLabel="% missing"
                color="#f59e0b"
                data={eda.missing.columns.filter((c) => c.missing > 0).slice(0, 15).map((c) => ({ name: c.column, value: c.percentage }))}
              />
            )}
            <DataTable
              columns={['Column', 'Missing', '% of rows']}
              rows={eda.missing.columns.map((c) => [c.column, int(c.missing), pct(c.percentage)])}
            />
          </>
        )}

        {tab === 'distributions' &&
          (distColumns.length === 0 ? (
            <Empty title="No numeric columns" />
          ) : (
            <>
              <div className="inline-controls">
                <label className="field inline">
                  <span>Column</span>
                  <select value={activeDist} onChange={(e) => setDistColumn(e.target.value)}>
                    {distColumns.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </label>
                <Badge tone="purple">{eda.distributions[activeDist].shape}</Badge>
                <span className="muted small">skewness {num(eda.distributions[activeDist].skewness)}</span>
              </div>
              <HistogramChart bins={eda.distributions[activeDist].bins} label={`Distribution of ${activeDist}`} />
            </>
          ))}

        {tab === 'correlations' &&
          (eda.correlations.columns.length < 2 ? (
            <Empty title="Not enough numeric columns" text="Correlation needs at least two numeric columns with variation." />
          ) : (
            <>
              <Heatmap columns={eda.correlations.columns} matrix={eda.correlations.matrix} />
              <DataTable
                columns={['Column A', 'Column B', 'Pearson r', 'Strength', 'Rows used']}
                rows={eda.correlations.top_pairs.map((p) => [
                  p.column_a, p.column_b, p.correlation.toFixed(3),
                  <Badge key="s" tone={Math.abs(p.correlation) >= 0.6 ? 'purple' : 'gray'}>{p.strength} {p.direction}</Badge>,
                  int(p.observations),
                ])}
              />
              <p className="muted small">Correlation does not imply causation.</p>
              <AiExplain datasetId={datasetId} kind="correlations" label="Explain these correlations" />
            </>
          ))}

        {tab === 'outliers' && (
          <DataTable
            empty="No numeric columns were large enough to check for outliers."
            columns={['Column', 'Outliers', '% of values', 'Lower bound', 'Upper bound', 'Most extreme']}
            rows={outlierRows.map(([c, o]) => [
              <strong key={c}>{c}</strong>,
              <Badge key="b" tone={o.outlier_count ? 'orange' : 'green'}>{int(o.outlier_count)}</Badge>,
              pct(o.outlier_percentage), num(o.lower_bound), num(o.upper_bound), num(o.max_outlier),
            ])}
          />
        )}

        {tab === 'insights' && (
          <DataTable
            empty="No column-level issues were detected."
            columns={['Column', 'Severity', 'Finding']}
            rows={eda.column_insights.map((i) => [
              <strong key="c">{i.column}</strong>,
              <Badge key="s" tone={i.severity === 'warning' ? 'orange' : 'blue'}>{i.severity}</Badge>,
              <span key="m" className="wrap">{i.message}</span>,
            ])}
          />
        )}
      </Panel>
    </>
  )
}
