import type { ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Loader2, X } from 'lucide-react'

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <div className="panel-heading">
          <div>
            {title && <h3>{title}</h3>}
            {subtitle && <p>{subtitle}</p>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  )
}

export function Stat({
  label,
  value,
  hint,
  tone = 'purple',
}: {
  label: string
  value: ReactNode
  hint?: string
  tone?: 'purple' | 'blue' | 'green' | 'orange' | 'red'
}) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <small>{hint}</small>}
    </div>
  )
}

export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="stat-grid">{children}</div>
}

export function Busy({ text = 'Loading…' }: { text?: string }) {
  return (
    <div className="loading-line">
      <Loader2 className="spin" size={17} />
      {text}
    </div>
  )
}

export function Empty({
  icon,
  title,
  text,
  action,
}: {
  icon?: ReactNode
  title: string
  text?: string
  action?: ReactNode
}) {
  return (
    <div className="empty-state compact">
      {icon && <div className="empty-icon">{icon}</div>}
      <h2>{title}</h2>
      {text && <p>{text}</p>}
      {action}
    </div>
  )
}

export function ErrorNote({
  message,
  onRetry,
  onClose,
}: {
  message: string
  onRetry?: () => void
  onClose?: () => void
}) {
  if (!message) return null
  return (
    <div className="alert alert-error" role="alert">
      <AlertTriangle size={16} />
      <span>{message}</span>
      {onRetry && (
        <button className="link-button" onClick={onRetry}>
          Retry
        </button>
      )}
      {onClose && (
        <button onClick={onClose} aria-label="Dismiss">
          <X size={16} />
        </button>
      )}
    </div>
  )
}

export function OkNote({ message }: { message: string }) {
  if (!message) return null
  return (
    <div className="alert alert-success">
      <CheckCircle2 size={17} />
      {message}
    </div>
  )
}

export function Badge({
  tone = 'gray',
  children,
}: {
  tone?: 'gray' | 'green' | 'orange' | 'red' | 'purple' | 'blue'
  children: ReactNode
}) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  )
}

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
}: {
  tabs: { id: T; label: string }[]
  value: T
  onChange: (id: T) => void
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={value === tab.id}
          className={`tab ${value === tab.id ? 'active' : ''}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

export function DataTable({
  columns,
  rows,
  empty = 'No data',
}: {
  columns: string[]
  rows: ReactNode[][]
  empty?: string
}) {
  if (rows.length === 0) return <p className="muted pad">{empty}</p>
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Renders the small markdown subset Gemini returns (bold, bullets) as React nodes. */
export function RichText({ text }: { text: string }) {
  const inline = (line: string) =>
    line.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
      part.startsWith('**') && part.endsWith('**') ? (
        <strong key={i}>{part.slice(2, -2)}</strong>
      ) : (
        <span key={i}>{part}</span>
      ),
    )
  const lines = text.split('\n').filter((l) => l.trim() !== '')
  const blocks: ReactNode[] = []
  let items: string[] = []
  const flush = () => {
    if (items.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`}>
          {items.map((it, i) => (
            <li key={i}>{inline(it)}</li>
          ))}
        </ul>,
      )
      items = []
    }
  }
  for (const raw of lines) {
    const line = raw.trim()
    const bullet = /^([*-]|\d+\.)\s+(.*)$/.exec(line)
    if (bullet) {
      items.push(bullet[2])
    } else {
      flush()
      blocks.push(<p key={`p-${blocks.length}`}>{inline(line.replace(/^#+\s*/, ''))}</p>)
    }
  }
  flush()
  return <div className="rich-text">{blocks}</div>
}
