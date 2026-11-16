import axios from 'axios'

export const API_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export const api = axios.create({ baseURL: API_URL })

/** Turn any API/network failure into one readable sentence. */
export function apiError(err: unknown, fallback = 'Something went wrong.'): string {
  if (axios.isAxiosError(err)) {
    if (!err.response) {
      return `Cannot reach the backend at ${API_URL}. Is it running?`
    }
    const detail = err.response.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item: { loc?: unknown[]; msg?: string }) =>
          `${item.loc ? item.loc.slice(1).join('.') + ': ' : ''}${item.msg ?? ''}`,
        )
        .join('; ')
    }
    if (detail && typeof detail === 'object' && 'message' in detail) {
      return String((detail as { message: unknown }).message)
    }
  }
  return fallback
}

export function downloadUrl(path: string): string {
  return `${API_URL}${path}`
}
