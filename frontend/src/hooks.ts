import { useCallback, useEffect, useEffectEvent, useState } from 'react'
import { apiError } from './lib/api'

export type AsyncState<T> = {
  data: T | null
  error: string
  loading: boolean
  reload: () => void
}

/**
 * Load data whenever `key` changes. `key` should be a primitive that identifies
 * the request (e.g. `${datasetId}:${chartType}`); pass null to skip loading.
 */
export function useAsync<T>(
  loader: () => Promise<T>,
  key: string | number | null,
): AsyncState<T> {
  const [state, setState] = useState<{ data: T | null; error: string; loading: boolean }>({
    data: null,
    error: '',
    loading: key !== null,
  })
  const [tick, setTick] = useState(0)
  const run = useEffectEvent(loader)

  useEffect(() => {
    if (key === null) return
    let cancelled = false
    const go = async () => {
      await Promise.resolve()
      if (cancelled) return
      setState((s) => ({ ...s, loading: true, error: '' }))
      try {
        const data = await run()
        if (!cancelled) setState({ data, error: '', loading: false })
      } catch (err) {
        if (!cancelled) setState({ data: null, error: apiError(err), loading: false })
      }
    }
    void go()
    return () => {
      cancelled = true
    }
  }, [key, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { ...state, reload }
}
