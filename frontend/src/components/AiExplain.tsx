import { useState } from 'react'
import { Sparkles } from 'lucide-react'
import { api, apiError } from '../lib/api'
import type { Grounding } from '../types'
import { Badge, Busy, ErrorNote, RichText } from './ui'

type Result = {
  explanation: string
  grounding: Grounding
  model: string
  latency_ms: number
}

/** "Explain with Gemini" button + result. Numbers are verified server-side. */
export function AiExplain({
  datasetId,
  kind,
  runId,
  chart,
  label = 'Explain with AI',
}: {
  datasetId: number
  kind: 'dataset' | 'quality' | 'eda' | 'correlations' | 'ml_run' | 'chart' | 'insights'
  runId?: number
  chart?: Record<string, unknown>
  label?: string
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<Result | null>(null)

  const run = async () => {
    setBusy(true)
    setError('')
    try {
      const response = await api.post(`/datasets/${datasetId}/ai/explain`, {
        kind,
        run_id: runId,
        chart,
      })
      setResult(response.data as Result)
    } catch (err) {
      setResult(null)
      setError(apiError(err, 'The AI explanation failed.'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="ai-explain">
      <button className="primary-button" onClick={() => void run()} disabled={busy}>
        <Sparkles size={15} />
        {busy ? 'Asking Gemini…' : label}
      </button>
      {busy && <Busy text="Gemini is reading the computed results…" />}
      <ErrorNote message={error} />
      {result && (
        <div className="ai-result">
          <div className="ai-result-head">
            <Badge tone={result.grounding.verified ? 'green' : 'orange'}>
              {result.grounding.verified
                ? `Numbers verified (${result.grounding.numbers_checked})`
                : `Unverified numbers: ${result.grounding.unverified_numbers.join(', ')}`}
            </Badge>
            <small className="muted">
              {result.model} · {(result.latency_ms / 1000).toFixed(1)}s
            </small>
          </div>
          <RichText text={result.explanation} />
          {!result.grounding.verified && (
            <p className="muted small">
              Some figures in this explanation could not be matched to computed results. Treat them
              with caution.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
