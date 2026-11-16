export function num(value?: number | null, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1000) {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)
  }
  return new Intl.NumberFormat(undefined, { maximumSignificantDigits: digits + 2 }).format(value)
}

export function int(value?: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return new Intl.NumberFormat().format(value)
}

export function pct(value?: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${value.toFixed(1)}%`
}

export function when(iso?: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso.endsWith('Z') ? iso : iso + 'Z')
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

export function bytes(n?: number | null): string {
  if (!n && n !== 0) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}
