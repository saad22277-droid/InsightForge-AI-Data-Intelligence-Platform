import { useEffect, useRef, useState } from 'react'
import { Bot, MessageSquarePlus, Send, Trash2, User } from 'lucide-react'
import { api, apiError } from '../lib/api'
import type { ChatMessage, Conversation, PageCtx } from '../types'
import { Badge, Busy, ErrorNote, RichText } from '../components/ui'
import { useAsync } from '../hooks'
import { when } from '../lib/format'

type AiStatus = { configured: boolean; model: string; free_tier_notice: string }

const SUGGESTIONS = [
  'Summarize this dataset in simple language.',
  'What are the most important findings in this dataset?',
  'Which columns have the most missing values?',
  'Find potential outliers.',
  'What relationships exist between the numeric columns?',
  'Which category performs best?',
]

export default function AskAiPage({ datasetId, fail }: PageCtx) {
  const status = useAsync(async () => (await api.get<AiStatus>('/ai/status')).data, 'status')
  const convs = useAsync(async () => (await api.get<Conversation[]>(`/datasets/${datasetId}/conversations`)).data, datasetId)

  const [activeId, setActiveId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const endRef = useRef<HTMLDivElement>(null)
  const tempId = useRef(0)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, sending])

  const open = async (id: number) => {
    setError('')
    try {
      const r = await api.get<Conversation>(`/datasets/${datasetId}/conversations/${id}`)
      setActiveId(id)
      setMessages(r.data.messages ?? [])
    } catch (err) {
      fail(apiError(err))
    }
  }

  const newChat = () => {
    setActiveId(null)
    setMessages([])
    setError('')
  }

  const remove = async (id: number) => {
    if (!window.confirm('Delete this conversation?')) return
    try {
      await api.delete(`/datasets/${datasetId}/conversations/${id}`)
      if (id === activeId) newChat()
      convs.reload()
    } catch (err) {
      fail(apiError(err))
    }
  }

  const send = async (text: string) => {
    const message = text.trim()
    if (!message || sending) return
    setSending(true)
    setError('')
    setInput('')
    tempId.current -= 1
    const temp: ChatMessage = { id: tempId.current, role: 'user', content: message, created_at: '' }
    setMessages((m) => [...m, temp])
    try {
      const r = await api.post<{ conversation_id: number; user_message: ChatMessage; assistant_message: ChatMessage }>(
        `/datasets/${datasetId}/chat`,
        { message, conversation_id: activeId ?? undefined },
      )
      setActiveId(r.data.conversation_id)
      setMessages((m) => [...m.filter((x) => x.id !== temp.id), r.data.user_message, r.data.assistant_message])
      convs.reload()
    } catch (err) {
      setMessages((m) => m.filter((x) => x.id !== temp.id))
      setInput(message)
      setError(apiError(err, 'The assistant could not answer.'))
    } finally {
      setSending(false)
    }
  }

  const unavailable = status.data && !status.data.configured

  return (
    <div className="chat-layout">
      <aside className="chat-sidebar panel">
        <button className="primary-button wide" onClick={newChat}><MessageSquarePlus size={15} /> New conversation</button>
        <div className="conv-list">
          {(convs.data ?? []).map((c) => (
            <div key={c.id} className={`conv-item ${c.id === activeId ? 'active' : ''}`}>
              <button onClick={() => void open(c.id)}>
                <strong>{c.title}</strong>
                <small>{c.message_count} messages · {when(c.updated_at)}</small>
              </button>
              <button className="icon-button small" aria-label="Delete conversation" onClick={() => void remove(c.id)}><Trash2 size={13} /></button>
            </div>
          ))}
          {convs.data?.length === 0 && <p className="muted small pad">Conversations are saved per dataset.</p>}
        </div>
      </aside>

      <section className="chat-main panel">
        {unavailable && (
          <div className="alert alert-warn">
            <span>Gemini is not configured. Add <code>GEMINI_API_KEY</code> to the backend <code>.env</code> file and restart the backend. Every other feature keeps working without it.</span>
          </div>
        )}
        <div className="chat-scroll">
          {messages.length === 0 && !sending && (
            <div className="chat-welcome">
              <Bot size={30} />
              <h2>Ask anything about this dataset</h2>
              <p>Answers are computed with validated analysis tools. Gemini only explains the results, and every number is checked against them.</p>
              <div className="chips">
                {SUGGESTIONS.map((s) => <button key={s} className="chip" disabled={!!unavailable} onClick={() => void send(s)}>{s}</button>)}
              </div>
            </div>
          )}
          {messages.map((m) => <Bubble key={m.id} message={m} />)}
          {sending && <div className="bubble assistant"><Busy text="Analysing the data and asking Gemini…" /></div>}
          <div ref={endRef} />
        </div>
        <ErrorNote message={error} onClose={() => setError('')} />
        <form className="chat-input" onSubmit={(e) => { e.preventDefault(); void send(input) }}>
          <textarea
            value={input}
            rows={2}
            maxLength={1000}
            placeholder="e.g. What is the average of a numeric column?"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(input) } }}
          />
          <button className="primary-button" type="submit" disabled={sending || !input.trim() || !!unavailable}><Send size={15} /> Send</button>
        </form>
        <small className="muted chat-foot">{status.data ? `Model: ${status.data.model}. ` : ''}Free-tier prompts may be used by Google; only computed aggregates and short value samples are sent. AI can make mistakes: check the evidence.</small>
      </section>
    </div>
  )
}

function Bubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user'
  const ev = (message.evidence ?? []).filter((e) => e.status === 'ok')
  const g = message.grounding
  return (
    <div className={`bubble ${message.role}`}>
      <div className="bubble-head">{isUser ? <User size={14} /> : <Bot size={14} />}<strong>{isUser ? 'You' : 'Assistant'}</strong></div>
      {isUser ? <p>{message.content}</p> : <RichText text={message.content} />}
      {!isUser && g && (
        <div className="bubble-meta">
          <Badge tone={g.verified ? 'green' : 'orange'}>{g.verified ? `Numbers verified (${g.numbers_checked})` : g.fallback ? 'Showing raw computed results' : 'Unverified numbers'}</Badge>
          {message.latency_ms ? <small className="muted">{(message.latency_ms / 1000).toFixed(1)}s</small> : null}
        </div>
      )}
      {ev.length > 0 && (
        <details className="evidence">
          <summary>How this was computed ({ev.length} analysis step{ev.length === 1 ? '' : 's'})</summary>
          {ev.map((e, i) => (
            <div key={i} className="evidence-item">
              <code>{e.tool}({Object.entries(e.args ?? {}).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : String(v)}`).join(', ')})</code>
              <pre>{JSON.stringify(e.data, null, 1)}</pre>
            </div>
          ))}
        </details>
      )}
    </div>
  )
}
