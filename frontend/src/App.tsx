import { useEffect, useMemo, useState, useCallback } from 'react'
import axios from 'axios'
import {
  Activity,
  BarChart3,
  Bot,
  BrainCircuit,
  Download,
  FileText,
  FlaskConical,
  LineChart as LineChartIcon,
  ScanSearch,
  Settings as SettingsIcon,
  Wand2,
  CheckCircle2,
  ChevronDown,
  Database,
  FileSpreadsheet,
  GitCompareArrows,
  LayoutDashboard,
  Loader2,
  Plus,
  RefreshCw,
  Trash2,
  ShieldCheck,
  Upload,
  X,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'
import { api, API_URL, downloadUrl } from './lib/api'
import type { PageCtx } from './types'
import AskAiPage from './pages/AskAiPage'
import CleaningPage from './pages/CleaningPage'
import EdaPage from './pages/EdaPage'
import ExperimentsPage from './pages/ExperimentsPage'
import MlPage from './pages/MlPage'
import ReportsPage from './pages/ReportsPage'
import SettingsPage from './pages/SettingsPage'
import VisualizationsPage from './pages/VisualizationsPage'

type Dataset = {
  id: number
  filename: string
  row_count?: number
  column_count?: number
  status?: string
  created_at?: string
}

type Summary = {
  row_count?: number
  column_count?: number
  duplicate_rows?: number
  columns?: string[]
  column_details?: Record<string, {
    data_type?: string
    missing_values?: number
    missing_percentage?: number
    unique_values?: number
  }>
  numeric_statistics?: Record<string, {
    mean?: number | null
    median?: number | null
    min?: number | null
    max?: number | null
    standard_deviation?: number | null
  }>
}

type Version = {
  id?: number
  version_number: number
  row_count?: number
  column_count?: number
  created_at?: string
}

type Page =
  | 'overview'
  | 'datasets'
  | 'versions'
  | 'cleaning'
  | 'quality'
  | 'eda'
  | 'visualizations'
  | 'ml'
  | 'ask'
  | 'reports'
  | 'experiments'
  | 'settings'

const LEGACY_PAGES: Page[] = ['overview', 'datasets', 'quality', 'versions']
const GLOBAL_PAGES: Page[] = ['experiments', 'settings']

function getArray<T>(data: unknown, key: string): T[] {
  if (Array.isArray(data)) return data as T[]

  if (
    data &&
    typeof data === 'object' &&
    Array.isArray((data as Record<string, unknown>)[key])
  ) {
    return (data as Record<string, unknown>)[key] as T[]
  }

  return []
}

function formatNumber(value?: number | null) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—'
  }

  return new Intl.NumberFormat().format(value)
}

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [quality, setQuality] = useState<Record<string, unknown> | null>(null)
  const [versions, setVersions] = useState<Version[]>([])
  const [comparison, setComparison] = useState<Record<string, unknown> | null>(null)
  const [fromVersion, setFromVersion] = useState('')
  const [toVersion, setToVersion] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)

  const selectedDataset = datasets.find((item) => item.id === selectedId)

  const loadDatasets = useCallback(async () => {
    try {
      const response = await api.get('/datasets')
      setError('')
      const items = getArray<Dataset>(response.data, 'datasets')
      setDatasets(items)

      if (items.length > 0) {
        setSelectedId((currentId) => currentId ?? items[0].id)
      }
    } catch {
      setError(
        'Could not connect to the backend. Make sure FastAPI is running at ' +
          API_URL,
      )
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const fetchDatasets = async () => {
      try {
        const response = await api.get('/datasets')
        if (cancelled) return

        setError('')
        const items = getArray<Dataset>(response.data, 'datasets')
        setDatasets(items)

        if (items.length > 0) {
          setSelectedId((currentId) => currentId ?? items[0].id)
        }
      } catch {
        if (!cancelled) {
          setError(
            'Could not connect to the backend. Make sure FastAPI is running at ' +
              API_URL,
          )
        }
      }
    }

    void fetchDatasets()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (selectedId === null) return

    let cancelled = false

    const fetchDatasetDetails = async () => {
      await Promise.resolve()
      if (cancelled) return

      setLoading(true)
      setError('')
      setSummary(null)
      setQuality(null)
      setVersions([])
      setComparison(null)

      try {
        const [summaryResponse, qualityResponse, versionsResponse] =
          await Promise.all([
            api.get(`/datasets/${selectedId}/summary`),
            api.get(`/datasets/${selectedId}/quality`),
            api.get(`/datasets/${selectedId}/versions`),
          ])

        if (cancelled) return

        const summaryData = summaryResponse.data as Record<string, unknown>
        setSummary((summaryData.summary || summaryData) as Summary)
        setQuality(qualityResponse.data as Record<string, unknown>)

        const versionItems = getArray<Version>(
          versionsResponse.data,
          'versions',
        )
        setVersions(versionItems)
      } catch {
        if (!cancelled) {
          setError(
            'Could not load all dataset details. Check that the backend endpoints are available.',
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void fetchDatasetDetails()

    return () => {
      cancelled = true
    }
  }, [selectedId, refreshKey])

  const handleClearWorkspace = async () => {
    if (!window.confirm(
      'Permanently delete all uploaded datasets, versions, CSV files, and reports?'
    )) return

    setError('')
    setNotice('')

    try {
      const response = await api.delete('/datasets/clear')

      setDatasets([])
      setSelectedId(null)
      setSummary(null)
      setQuality(null)
      setVersions([])
      setComparison(null)
      setFromVersion('')
      setToVersion('')
      setPage('overview')

      setNotice(response.data.message || 'Workspace cleared successfully.')
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(
          err.response?.data?.detail || 'Could not clear the workspace.'
        )
      } else {
        setError('Could not clear the workspace.')
      }
    }
  }

  const handleUpload = async (file?: File) => {
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a CSV file.')
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    setUploading(true)
    setError('')
    setNotice('')

    try {
      const response = await api.post('/datasets/upload', formData)
      const newId = response.data.id

      setNotice('Dataset uploaded successfully.')

      const listResponse = await api.get('/datasets')
      const items = getArray<Dataset>(listResponse.data, 'datasets')
      setDatasets(items)

      if (typeof newId === 'number') {
        setSelectedId(newId)
      } else if (items.length > 0) {
        setSelectedId(items[items.length - 1].id)
      }

      setPage('overview')
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(
          err.response?.data?.detail ||
            'Dataset upload failed. Check the backend and try again.',
        )
      } else {
        setError('Dataset upload failed.')
      }
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (id: number, name: string) => {
    if (!window.confirm(`Delete "${name}" with all its versions, models, reports and conversations?`)) return
    setError('')
    try {
      await api.delete(`/datasets/${id}`)
      setNotice(`Deleted ${name}.`)
      if (id === selectedId) setSelectedId(null)
      await loadDatasets()
    } catch (err) {
      setError(
        axios.isAxiosError(err) && typeof err.response?.data?.detail === 'string'
          ? err.response.data.detail
          : 'Could not delete the dataset.',
      )
    }
  }

  const handleVersionUpload = async (file?: File) => {
    if (!file || !selectedId) return
    const formData = new FormData()
    formData.append('file', file)
    setUploading(true)
    setError('')
    try {
      await api.post(`/datasets/${selectedId}/versions/upload`, formData)
      setNotice('New dataset version uploaded.')
      await loadDatasets()
      setRefreshKey((k) => k + 1)
    } catch (err) {
      setError(
        axios.isAxiosError(err) && typeof err.response?.data?.detail === 'string'
          ? err.response.data.detail
          : 'Version upload failed.',
      )
    } finally {
      setUploading(false)
    }
  }

  const handleCompare = async () => {
    if (!selectedId || !fromVersion || !toVersion) return

    try {
      setError('')
      setComparison(null)

      const response = await api.get(
        `/datasets/${selectedId}/versions/compare`,
        {
          params: {
            from: fromVersion,
            to: toVersion,
          },
        },
      )

      setComparison(response.data)
    } catch {
      setError('Version comparison failed. Check the selected versions.')
    }
  }

  const chartData = useMemo(() => {
    if (!summary?.numeric_statistics) return []

    return Object.entries(summary.numeric_statistics)
      .filter(([, stats]) => typeof stats.mean === 'number')
      .map(([name, stats]) => ({
        name,
        mean: stats.mean as number,
      }))
  }, [summary])

  const navigation = [
    { id: 'overview' as const, label: 'Overview', icon: LayoutDashboard },
    { id: 'datasets' as const, label: 'Datasets', icon: Database },
    { id: 'versions' as const, label: 'Versions', icon: GitCompareArrows },
    { id: 'cleaning' as const, label: 'Data Cleaning', icon: Wand2 },
    { id: 'quality' as const, label: 'Data Quality', icon: ShieldCheck },
    { id: 'eda' as const, label: 'Exploratory Analysis', icon: ScanSearch },
    { id: 'visualizations' as const, label: 'Visualizations', icon: LineChartIcon },
    { id: 'ml' as const, label: 'Machine Learning', icon: BrainCircuit },
    { id: 'ask' as const, label: 'Ask AI', icon: Bot },
    { id: 'reports' as const, label: 'Reports', icon: FileText },
    { id: 'experiments' as const, label: 'Experiments', icon: FlaskConical },
    { id: 'settings' as const, label: 'Settings', icon: SettingsIcon },
  ]

  const legacyPage = LEGACY_PAGES.includes(page)
  const globalPage = GLOBAL_PAGES.includes(page)
  const ctx: PageCtx | null = selectedDataset
    ? {
        datasetId: selectedDataset.id,
        filename: selectedDataset.filename,
        notify: setNotice,
        fail: setError,
        refresh: () => {
          void loadDatasets()
          setRefreshKey((k) => k + 1)
        },
      }
    : null

  const title = navigation.find((item) => item.id === page)?.label

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={23} />
          </div>
          <div>
            <strong>Data Intelligence</strong>
            <span>ANALYTICS PLATFORM</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => setPage(id)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="backend-indicator">
            <span className="status-dot" />
            <div>
              <strong>API connection</strong>
              <small>FastAPI backend</small>
            </div>
          </div>
          <small className="sidebar-version">AI Data Platform · v1.0</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace / {title}</div>
            <h1>{title}</h1>
          </div>

          <div className="topbar-actions">
            <button
              className="icon-button"
              title="Refresh datasets"
              onClick={() => void loadDatasets()}
            >
              <RefreshCw size={18} />
            </button>
            <button
              className="icon-button"
              title="Clear workspace"
              aria-label="Clear workspace"
              onClick={() => void handleClearWorkspace()}
            >
              <Trash2 size={18} />
            </button>
            <label className="upload-button">
              <Plus size={17} />
              <span>Upload dataset</span>
              <input
                type="file"
                accept=".csv"
                hidden
                onChange={(event) => {
                  void handleUpload(event.target.files?.[0])
                  event.target.value = ''
                }}
              />
            </label>
          </div>
        </header>

        {error && (
          <div className="alert alert-error">
            <span>{error}</span>
            <button onClick={() => setError('')} aria-label="Dismiss">
              <X size={16} />
            </button>
          </div>
        )}

        {notice && (
          <div className="alert alert-success">
            <CheckCircle2 size={17} />
            {notice}
            <button onClick={() => setNotice('')} aria-label="Dismiss">
              <X size={16} />
            </button>
          </div>
        )}

        {uploading && (
          <div className="loading-line">
            <Loader2 className="spin" size={17} />
            Uploading and profiling dataset...
          </div>
        )}

        <section className="dataset-toolbar">
          <div>
            <span className="eyebrow">CURRENT DATASET</span>
            <div className="dataset-select-wrap">
              <FileSpreadsheet size={19} />
              <select
                value={selectedId ?? ''}
                onChange={(event) =>
                  setSelectedId(event.target.value ? Number(event.target.value) : null)
                }
              >
                <option value="">Select a dataset</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.filename}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} />
            </div>
          </div>
          <span className="dataset-count">
            {datasets.length} dataset{datasets.length === 1 ? '' : 's'} available
          </span>
        </section>

        {globalPage ? (
          page === 'settings' ? <SettingsPage /> : <ExperimentsPage />
        ) : loading ? (
          <div className="empty-state">
            <Loader2 className="spin" size={28} />
            <p>Loading dataset analytics...</p>
          </div>
        ) : !selectedDataset ? (
          <div className="empty-state">
            <div className="empty-icon">
              <Upload size={28} />
            </div>
            <h2>Start with your first dataset</h2>
            <p>Upload a CSV file to explore your data and generate analytics.</p>
            <label className="upload-button">
              <Upload size={17} />
              Upload CSV
              <input
                type="file"
                accept=".csv"
                hidden
                onChange={(event) => {
                  void handleUpload(event.target.files?.[0])
                  event.target.value = ''
                }}
              />
            </label>
          </div>
        ) : (
          <>
            {legacyPage && (<>
            <section className="welcome-row">
              <div>
                <h2>{selectedDataset.filename}</h2>
                <p>Your dataset analytics and insights at a glance.</p>
              </div>
              <span className="status-pill">
                <span className="status-dot" />
                {selectedDataset.status || 'Available'}
              </span>
            </section>

            <section className="metric-grid">
              <MetricCard
                label="Total rows"
                value={formatNumber(summary?.row_count ?? selectedDataset.row_count)}
                icon={<Database size={19} />}
                tone="blue"
              />
              <MetricCard
                label="Total columns"
                value={formatNumber(summary?.column_count ?? selectedDataset.column_count)}
                icon={<BarChart3 size={19} />}
                tone="purple"
              />
              <MetricCard
                label="Duplicate rows"
                value={formatNumber(summary?.duplicate_rows)}
                icon={<GitCompareArrows size={19} />}
                tone="orange"
              />
              <MetricCard
                label="Versions"
                value={formatNumber(versions.length)}
                icon={<Activity size={19} />}
                tone="green"
              />
            </section>
            </>)}

            {page === 'overview' && (
              <>
                <section className="content-grid">
                  <div className="panel chart-panel">
                    <div className="panel-heading">
                      <div>
                        <h3>Numeric column averages</h3>
                        <p>Mean values across numeric columns</p>
                      </div>
                      <BarChart3 size={19} />
                    </div>

                    {chartData.length > 0 ? (
                      <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                            <YAxis tick={{ fontSize: 11 }} />
                            <Tooltip />
                            <Bar
                              dataKey="mean"
                              fill="#6558e8"
                              radius={[5, 5, 0, 0]}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="chart-empty">
                        Numeric summary data will appear here when available.
                      </div>
                    )}
                  </div>

                  <div className="panel">
                    <div className="panel-heading">
                      <div>
                        <h3>Dataset information</h3>
                        <p>Column-level overview</p>
                      </div>
                      <FileSpreadsheet size={19} />
                    </div>
                    <div className="column-list">
                      {(summary?.columns || []).slice(0, 7).map((column) => {
                        const details = summary?.column_details?.[column]
                        return (
                          <div className="column-row" key={column}>
                            <div>
                              <strong>{column}</strong>
                              <small>{details?.data_type || 'Column'}</small>
                            </div>
                            <span>
                              {formatNumber(details?.unique_values)} unique
                            </span>
                          </div>
                        )
                      })}
                      {!summary?.columns?.length && (
                        <p className="muted">No column information available.</p>
                      )}
                    </div>
                  </div>
                </section>

                <section className="panel table-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Missing values by column</h3>
                      <p>Completeness overview</p>
                    </div>
                  </div>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Data type</th>
                          <th>Missing values</th>
                          <th>Missing percentage</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(summary?.column_details || {}).map(
                          ([column, details]) => (
                            <tr key={column}>
                              <td>{column}</td>
                              <td>{details.data_type || '—'}</td>
                              <td>{formatNumber(details.missing_values)}</td>
                              <td>
                                {details.missing_percentage === undefined
                                  ? '—'
                                  : `${details.missing_percentage.toFixed(2)}%`}
                              </td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                  </div>
                </section>
              </>
            )}

            {page === 'datasets' && (
              <section className="panel table-panel">
                <div className="panel-heading">
                  <div>
                    <h3>All datasets</h3>
                    <p>Uploaded datasets in your workspace</p>
                  </div>
                  <label className="ghost-button">
                    <Plus size={14} /> Upload new version of {selectedDataset.filename}
                    <input
                      type="file"
                      accept=".csv"
                      hidden
                      onChange={(event) => {
                        void handleVersionUpload(event.target.files?.[0])
                        event.target.value = ''
                      }}
                    />
                  </label>
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Dataset</th>
                        <th>Rows</th>
                        <th>Columns</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {datasets.map((dataset) => (
                        <tr
                          key={dataset.id}
                          className="clickable-row"
                          onClick={() => {
                            setSelectedId(dataset.id)
                            setPage('overview')
                          }}
                        >
                          <td>{dataset.filename}</td>
                          <td>{formatNumber(dataset.row_count)}</td>
                          <td>{formatNumber(dataset.column_count)}</td>
                          <td>{dataset.status || 'Available'}</td>
                          <td onClick={(event) => event.stopPropagation()}>
                            <div className="button-row">
                              <a
                                className="ghost-button"
                                href={downloadUrl(`/datasets/${dataset.id}/export`)}
                                title="Download active version as CSV"
                              >
                                <Download size={13} /> CSV
                              </a>
                              <button
                                className="icon-button small"
                                aria-label={`Delete ${dataset.filename}`}
                                onClick={() => void handleDelete(dataset.id, dataset.filename)}
                              >
                                <Trash2 size={14} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {page === 'quality' && (
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h3>Data quality report</h3>
                    <p>Quality metrics returned by the backend</p>
                  </div>
                  <ShieldCheck size={20} />
                </div>
                {quality ? (
                  <pre className="json-view">
                    {JSON.stringify(quality, null, 2)}
                  </pre>
                ) : (
                  <p className="muted">No quality report is available.</p>
                )}
              </section>
            )}

            {page === 'versions' && (
              <>
                <section className="panel table-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Dataset version history</h3>
                      <p>Uploaded versions of this dataset</p>
                    </div>
                  </div>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Version</th>
                          <th>Rows</th>
                          <th>Columns</th>
                          <th>Created</th>
                        </tr>
                      </thead>
                      <tbody>
                        {versions.map((version) => (
                          <tr key={version.id ?? version.version_number}>
                            <td>Version {version.version_number}</td>
                            <td>{formatNumber(version.row_count)}</td>
                            <td>{formatNumber(version.column_count)}</td>
                            <td>
                              {version.created_at
                                ? new Date(version.created_at).toLocaleString()
                                : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="panel compare-panel">
                  <div className="panel-heading">
                    <div>
                      <h3>Compare versions</h3>
                      <p>Review changes between two dataset versions</p>
                    </div>
                    <GitCompareArrows size={20} />
                  </div>
                  <div className="compare-controls">
                    <label>
                      From version
                      <select
                        value={fromVersion}
                        onChange={(event) => setFromVersion(event.target.value)}
                      >
                        <option value="">Select version</option>
                        {versions.map((version) => (
                          <option
                            key={version.version_number}
                            value={version.version_number}
                          >
                            Version {version.version_number}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      To version
                      <select
                        value={toVersion}
                        onChange={(event) => setToVersion(event.target.value)}
                      >
                        <option value="">Select version</option>
                        {versions.map((version) => (
                          <option
                            key={version.version_number}
                            value={version.version_number}
                          >
                            Version {version.version_number}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      className="primary-button"
                      disabled={!fromVersion || !toVersion || fromVersion === toVersion}
                      onClick={() => void handleCompare()}
                    >
                      Compare
                    </button>
                  </div>
                  {comparison && (
                    <pre className="json-view">
                      {JSON.stringify(comparison, null, 2)}
                    </pre>
                  )}
                </section>
              </>
            )}
            {ctx && page === 'cleaning' && <CleaningPage key={`c${selectedId}`} {...ctx} />}
            {ctx && page === 'eda' && <EdaPage key={`e${selectedId}${refreshKey}`} {...ctx} />}
            {ctx && page === 'visualizations' && <VisualizationsPage key={`v${selectedId}${refreshKey}`} {...ctx} />}
            {ctx && page === 'ml' && <MlPage key={`m${selectedId}${refreshKey}`} {...ctx} />}
            {ctx && page === 'ask' && <AskAiPage key={`a${selectedId}`} {...ctx} />}
            {ctx && page === 'reports' && <ReportsPage key={`r${selectedId}${refreshKey}`} {...ctx} />}
          </>
        )}

        <footer className="footer">
          <span>AI Data Intelligence Platform</span>
          <span>FastAPI · React · PostgreSQL · scikit-learn · Google Gemini</span>
        </footer>
      </main>
    </div>
  )
}

function MetricCard({
  label,
  value,
  icon,
  tone,
}: {
  label: string
  value: string
  icon: React.ReactNode
  tone: string
}) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${tone}`}>{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default App




