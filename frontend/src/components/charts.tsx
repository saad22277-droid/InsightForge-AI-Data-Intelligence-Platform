import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'

const COLORS = ['#6558e8', '#f59e0b', '#10b981', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6', '#8b5cf6']
const AXIS = { fontSize: 11, fill: '#777991' }

const fmt = (v: number) =>
  Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : String(+v.toPrecision(4))) : ''

export type Bin = { start: number; end: number; count: number }

export function HistogramChart({ bins, label }: { bins: Bin[]; label?: string }) {
  const data = bins.map((b) => ({ range: `${fmt(b.start)}–${fmt(b.end)}`, count: b.count }))
  return (
    <div className="chart-container" aria-label={label}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="range" tick={AXIS} interval="preserveStartEnd" />
          <YAxis tick={AXIS} allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill={COLORS[0]} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function CategoryBars({
  data,
  horizontal = false,
  valueLabel = 'value',
  color = COLORS[0],
}: {
  data: { name: string; value: number }[]
  horizontal?: boolean
  valueLabel?: string
  color?: string
}) {
  return (
    <div className="chart-container" style={{ height: horizontal ? Math.max(245, data.length * 28) : 245 }}>
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout={horizontal ? 'vertical' : 'horizontal'}
          margin={{ top: 8, right: 16, left: horizontal ? 40 : 0, bottom: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={horizontal} horizontal={!horizontal} />
          {horizontal ? (
            <>
              <XAxis type="number" tick={AXIS} />
              <YAxis type="category" dataKey="name" tick={AXIS} width={110} />
            </>
          ) : (
            <>
              <XAxis dataKey="name" tick={AXIS} interval={0} angle={data.length > 6 ? -25 : 0} textAnchor={data.length > 6 ? 'end' : 'middle'} height={data.length > 6 ? 60 : 30} />
              <YAxis tick={AXIS} />
            </>
          )}
          <Tooltip formatter={(v) => [fmt(Number(v)), valueLabel]} />
          <Bar dataKey="value" fill={color} radius={horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function LineSeries({
  data,
  xLabel,
  yLabel,
}: {
  data: { x: string | number; y: number }[]
  xLabel: string
  yLabel: string
}) {
  return (
    <div className="chart-container">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" tick={AXIS} label={{ value: xLabel, position: 'insideBottom', offset: -10, fontSize: 11 }} />
          <YAxis tick={AXIS} />
          <Tooltip formatter={(v) => [fmt(Number(v)), yLabel]} />
          <Line type="monotone" dataKey="y" stroke={COLORS[0]} dot={data.length < 40} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export function ScatterView({
  series,
  xLabel,
  yLabel,
  diagonal = false,
}: {
  series: { name: string; data: { x: number; y: number }[] }[]
  xLabel: string
  yLabel: string
  diagonal?: boolean
}) {
  return (
    <div className="chart-container tall">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="x" name={xLabel} tick={AXIS} domain={['auto', 'auto']} label={{ value: xLabel, position: 'insideBottom', offset: -10, fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name={yLabel} tick={AXIS} domain={['auto', 'auto']} />
          <ZAxis range={[28, 28]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          {series.length > 1 && <Legend />}
          {series.map((s, i) => (
            <Scatter key={s.name} name={s.name} data={s.data} fill={COLORS[i % COLORS.length]} fillOpacity={0.65} />
          ))}
          {diagonal && null}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

export type BoxStat = {
  label: string; n: number; min: number; q1: number; median: number; q3: number
  max: number; mean: number; outlier_count: number; outliers: number[]
}

/** Box-and-whisker plot drawn in SVG from real quartiles returned by the API. */
export function BoxPlot({ boxes }: { boxes: BoxStat[] }) {
  const W = 640
  const H = 300
  const pad = { l: 56, r: 16, t: 14, b: 56 }
  const all = boxes.flatMap((b) => [b.min, b.max, ...b.outliers])
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  const span = hi - lo || 1
  const y = (v: number) => pad.t + (1 - (v - lo) / span) * (H - pad.t - pad.b)
  const slot = (W - pad.l - pad.r) / boxes.length
  const ticks = Array.from({ length: 5 }, (_, i) => lo + (span * i) / 4)
  return (
    <div className="chart-container tall box-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Box plot" width="100%" height="100%">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#eceef4" />
            <text x={pad.l - 6} y={y(t) + 4} fontSize="10" fill="#777991" textAnchor="end">{fmt(t)}</text>
          </g>
        ))}
        {boxes.map((b, i) => {
          const cx = pad.l + slot * i + slot / 2
          const w = Math.min(54, slot * 0.55)
          const color = COLORS[i % COLORS.length]
          return (
            <g key={b.label}>
              <line x1={cx} x2={cx} y1={y(b.min)} y2={y(b.q1)} stroke={color} strokeWidth="1.5" />
              <line x1={cx} x2={cx} y1={y(b.q3)} y2={y(b.max)} stroke={color} strokeWidth="1.5" />
              <line x1={cx - w / 4} x2={cx + w / 4} y1={y(b.min)} y2={y(b.min)} stroke={color} strokeWidth="1.5" />
              <line x1={cx - w / 4} x2={cx + w / 4} y1={y(b.max)} y2={y(b.max)} stroke={color} strokeWidth="1.5" />
              <rect x={cx - w / 2} y={y(b.q3)} width={w} height={Math.max(1, y(b.q1) - y(b.q3))} fill={color} fillOpacity="0.22" stroke={color} strokeWidth="1.5" rx="3">
                <title>{`${b.label}\nn=${b.n}\nmedian ${fmt(b.median)}\nQ1 ${fmt(b.q1)} · Q3 ${fmt(b.q3)}\noutliers: ${b.outlier_count}`}</title>
              </rect>
              <line x1={cx - w / 2} x2={cx + w / 2} y1={y(b.median)} y2={y(b.median)} stroke="#202338" strokeWidth="2" />
              {b.outliers.map((o, k) => (
                <circle key={k} cx={cx} cy={y(o)} r="2.5" fill="none" stroke="#ef4444" />
              ))}
              <text x={cx} y={H - pad.b + 16} fontSize="10" fill="#55576b" textAnchor="middle">
                {b.label.length > 14 ? b.label.slice(0, 13) + '…' : b.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/** Correlation heat-map rendered as a CSS grid (diverging blue ↔ red). */
export function Heatmap({ columns, matrix }: { columns: string[]; matrix: (number | null)[][] }) {
  const color = (v: number | null) => {
    if (v === null) return '#f3f4f8'
    const a = Math.min(1, Math.abs(v))
    return v >= 0 ? `rgba(101, 88, 232, ${0.08 + a * 0.8})` : `rgba(239, 68, 68, ${0.08 + a * 0.8})`
  }
  return (
    <div className="heatmap-wrap">
      <div className="heatmap" style={{ gridTemplateColumns: `110px repeat(${columns.length}, minmax(46px, 1fr))` }}>
        <span />
        {columns.map((c) => (
          <span key={c} className="heat-head" title={c}>{c.length > 9 ? c.slice(0, 8) + '…' : c}</span>
        ))}
        {matrix.map((row, i) => (
          <HeatRow key={columns[i]} name={columns[i]} row={row} color={color} />
        ))}
      </div>
      <div className="heat-legend"><span>−1</span><i /><span>+1</span></div>
    </div>
  )
}

function HeatRow({ name, row, color }: { name: string; row: (number | null)[]; color: (v: number | null) => string }) {
  return (
    <>
      <span className="heat-side" title={name}>{name.length > 14 ? name.slice(0, 13) + '…' : name}</span>
      {row.map((v, j) => (
        <span key={j} className="heat-cell" style={{ background: color(v), color: v !== null && Math.abs(v) > 0.55 ? '#fff' : '#33354a' }} title={`${name}: ${v ?? 'n/a'}`}>
          {v === null ? '–' : v.toFixed(2)}
        </span>
      ))}
    </>
  )
}
