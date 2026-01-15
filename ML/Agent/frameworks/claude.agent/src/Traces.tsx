import { useState, useEffect, useRef } from 'react'
import './Traces.css'
import SmartTraces, { TraceEvent } from './SmartTraces'

type TraceView = 'raw' | 'smart'

const API_BASE_URL = 'http://localhost:3001/api/traces'

function Traces() {
  const [view, setView] = useState<TraceView>('smart')
  const [traces, setTraces] = useState<TraceEvent[]>([])
  const [filter, setFilter] = useState<string>('')
  const [autoScroll, setAutoScroll] = useState(true)
  const tracesEndRef = useRef<HTMLDivElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    // Connect to SSE endpoint
    const eventSource = new EventSource(`${API_BASE_URL}/stream`)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      const trace: TraceEvent = JSON.parse(event.data)
      setTraces((prev) => [...prev, trace])
    }

    eventSource.onerror = () => {
      console.error('SSE connection error')
    }

    // Load existing history
    fetch(`${API_BASE_URL}/history`)
      .then((res) => res.json())
      .then((data) => {
        if (data.history) {
          setTraces(data.history)
        }
      })
      .catch(console.error)

    return () => {
      eventSource.close()
    }
  }, [])

  useEffect(() => {
    if (autoScroll && tracesEndRef.current) {
      tracesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [traces, autoScroll])

  const clearTraces = async () => {
    try {
      await fetch(`${API_BASE_URL}/clear`, { method: 'POST' })
      setTraces([])
    } catch (error) {
      console.error('Failed to clear traces:', error)
    }
  }

  const filteredTraces = filter
    ? traces.filter(
        (t) =>
          t.category.toLowerCase().includes(filter.toLowerCase()) ||
          (t.sessionId && t.sessionId.toLowerCase().includes(filter.toLowerCase()))
      )
    : traces

  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      USER_MESSAGE: '#4CAF50',
      API_REQUEST: '#2196F3',
      API_RESPONSE: '#FF9800',
      ASSISTANT_RESPONSE: '#9C27B0',
    }
    return colors[category] || '#757575'
  }

  // Render Smart View
  if (view === 'smart') {
    return (
      <div className="traces-container">
        <div className="traces-header">
          <h1>Agent Traces</h1>
          <p>Real-time monitoring of agent activity and API communications</p>

          <div className="view-toggle">
            <button
              className={`view-button ${view === 'smart' ? 'active' : ''}`}
              onClick={() => setView('smart')}
            >
              Smart View
            </button>
            <button
              className={`view-button ${view === 'raw' ? 'active' : ''}`}
              onClick={() => setView('raw')}
            >
              Raw Traces
            </button>
          </div>
        </div>

        <SmartTraces traces={traces} onClear={clearTraces} />
      </div>
    )
  }

  // Render Raw View
  return (
    <div className="traces-container">
      <div className="traces-header">
        <h1>Agent Traces</h1>
        <p>Real-time monitoring of agent activity and API communications</p>

        <div className="view-toggle">
          <button
            className={`view-button ${view === 'smart' ? 'active' : ''}`}
            onClick={() => setView('smart')}
          >
            Smart View
          </button>
          <button
            className={`view-button ${view === 'raw' ? 'active' : ''}`}
            onClick={() => setView('raw')}
          >
            Raw Traces
          </button>
        </div>
      </div>

      <div className="traces-controls">
        <input
          type="text"
          className="filter-input"
          placeholder="Filter by category or session..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <label className="auto-scroll-toggle">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          Auto-scroll
        </label>
        <button className="clear-button" onClick={clearTraces}>
          Clear Traces
        </button>
        <div className="trace-count">
          {filteredTraces.length} trace{filteredTraces.length !== 1 ? 's' : ''}
        </div>
      </div>

      <div className="traces-list">
        {filteredTraces.length === 0 ? (
          <div className="empty-state">
            <p>No traces yet. Start a conversation in the Chat tab to see agent activity.</p>
          </div>
        ) : (
          filteredTraces.map((trace, idx) => (
            <div key={idx} className="trace-item">
              <div className="trace-header-row">
                <span
                  className="trace-category"
                  style={{ backgroundColor: getCategoryColor(trace.category) }}
                >
                  {trace.category}
                </span>
                <span className="trace-timestamp">{new Date(trace.timestamp).toLocaleTimeString()}</span>
                {trace.sessionId && <span className="trace-session">Session: {trace.sessionId}</span>}
              </div>
              <pre className="trace-data">{JSON.stringify(trace.data, null, 2)}</pre>
            </div>
          ))
        )}
        <div ref={tracesEndRef} />
      </div>
    </div>
  )
}

export default Traces
