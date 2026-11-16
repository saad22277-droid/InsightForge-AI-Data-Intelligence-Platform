export type PageCtx = {
  datasetId: number
  filename: string
  notify: (message: string) => void
  fail: (message: string) => void
  refresh: () => void
}

export type ColumnInfo = { name: string; dtype: string; missing: number; unique: number }

export type Metadata = {
  id: number
  filename: string
  row_count: number
  column_count: number
  version_count: number
  active_version?: number | null
  size_bytes: number | null
  sha256: string | null
  created_at: string | null
  columns: ColumnInfo[]
}

export type NumStats = {
  count: number; missing: number; mean: number | null; std: number | null
  min: number | null; q1: number | null; median: number | null; q3: number | null
  max: number | null; skewness: number | null; unique: number
}

export type Eda = {
  overview: {
    row_count: number; column_count: number; memory_bytes: number
    numeric_columns: string[]; categorical_columns: string[]
    datetime_like_columns: string[]; dtypes: Record<string, string>
  }
  numeric_statistics: Record<string, NumStats>
  categorical_statistics: Record<string, {
    count: number; missing: number; unique: number; mode: string | null
    top_values: { value: string; count: number; percentage: number }[]
  }>
  missing: {
    total_missing_cells: number; rows_with_missing: number; complete_rows: number
    columns: { column: string; missing: number; percentage: number }[]
  }
  duplicates: { duplicate_rows: number; percentage: number }
  distributions: Record<string, {
    bins: { start: number; end: number; count: number }[]
    skewness: number | null; shape: string
  }>
  correlations: {
    columns: string[]; matrix: (number | null)[][]
    top_pairs: { column_a: string; column_b: string; correlation: number
      strength: string; direction: string; observations: number }[]
  }
  outliers: {
    total_outliers: number
    columns: Record<string, { outlier_count: number; outlier_percentage: number
      lower_bound: number; upper_bound: number; max_outlier: number | null }>
  }
  column_insights: { column: string; severity: string; type: string; message: string }[]
  quality: {
    quality_score: number
    score_components: Record<string, number>
    warnings: string[]
    recommendations: string[]
  }
}

export type Grounding = {
  verified: boolean
  numbers_checked: number
  unverified_numbers: string[]
  fallback?: boolean
}

export type Recommendation = {
  id: string; category: string; severity: string; column: string | null
  title: string; reason: string; estimated_impact: string | null
  operation: Record<string, unknown>
}

export type Snapshot = {
  rows: number; columns: number; missing_cells: number; duplicate_rows: number
  dtypes: Record<string, string>
}

export type CleaningPreview = {
  before: Snapshot; after: Snapshot
  delta: { rows: number; columns: number; missing_cells: number; duplicate_rows: number }
  dtype_changes: { column: string; from: string; to: string }[]
  missing_value_changes: { column: string; from: number; to: number }[]
  numeric_changes: { column: string
    before: Record<string, number | null>; after: Record<string, number | null> }[]
  operations: { type: string; message: string; rows_affected: number }[]
  warnings: string[]
  sample_before: Record<string, unknown>[]
  sample_after: Record<string, unknown>[]
}

export type MlRun = {
  id: number; dataset_id: number; dataset_filename?: string
  dataset_version_number: number | null; dataset_sha256: string | null
  model_name: string; version: number; task: string; algorithm: string
  target: string | null; features: string[]; config: Record<string, unknown>
  metrics: Record<string, number>; primary_metric: string
  primary_metric_value: number | null; status: string; duration_ms: number
  environment: Record<string, string> | null; created_at: string
  details?: {
    warnings?: string[]
    excluded_features?: { column: string; reason: string }[]
    feature_importance?: { feature: string; importance: number }[]
    split?: { train_rows: number; test_rows: number; test_size: number }
    confusion_matrix?: number[][]
    classes?: string[]
    predictions_sample?: { actual: number; predicted: number }[]
    cluster_profiles?: { cluster: number; size: number; percentage: number
      numeric_means: Record<string, number> }[]
    pca_projection?: { x: number; y: number; cluster: number }[]
    note?: string
  }
}

export type FeatureSchema = {
  name: string; kind: 'numeric' | 'categorical'
  min?: number; max?: number; mean?: number; median?: number; values?: string[]
}

export type ChatMessage = {
  id: number; role: 'user' | 'assistant'; content: string
  evidence?: { tool: string; status: string; args?: Record<string, unknown>
    data?: unknown; error?: string }[] | null
  grounding?: Grounding | null
  model?: string | null; latency_ms?: number | null; created_at: string
}

export type Conversation = {
  id: number; title: string; updated_at: string; message_count?: number
  messages?: ChatMessage[]
}

export type ReportMeta = {
  id: number; title: string; size_bytes: number; include_ai: boolean
  ai_included: boolean; created_at: string; download_url: string
  sections?: { ai_notes?: string[]; models?: number } | null
}
