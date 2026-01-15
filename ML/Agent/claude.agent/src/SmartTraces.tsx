import { useState, useEffect, useRef } from 'react'
import './SmartTraces.css'

export interface TraceEvent {
  timestamp: string
  category: string
  data: any
  sessionId?: string
}

interface ConversationFlow {
  timestamp: string
  type: 'user' | 'api_call' | 'response' | 'tool_call' | 'tool_result'
  userMessage?: string
  apiMetadata?: {
    model: string
    maxTokens: number
    duration?: number
    toolRound?: number
  }
  responseMetadata?: {
    id: string
    model: string
    stopReason: string
    inputTokens: number
    outputTokens: number
    duration: number
  }
  assistantMessage?: string
  toolCall?: {
    toolName: string
    toolInput: any
    toolUseId: string
  }
  toolResult?: {
    toolName: string
    result: any
    success: boolean
    error?: string
  }
}

interface SmartTracesProps {
  traces: TraceEvent[]
  onClear: () => void
}

function SmartTraces({ traces, onClear }: SmartTracesProps) {
  const [conversationFlow, setConversationFlow] = useState<ConversationFlow[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const tracesEndRef = useRef<HTMLDivElement>(null)

  // Process traces into conversation flow
  useEffect(() => {
    if (!traces) return

    const flow: ConversationFlow[] = []

    for (let i = 0; i < traces.length; i++) {
      const trace = traces[i]

      if (trace.category === 'USER_MESSAGE') {
        flow.push({
          timestamp: trace.timestamp,
          type: 'user',
          userMessage: trace.data.message,
        })
      } else if (trace.category === 'API_REQUEST') {
        flow.push({
          timestamp: trace.timestamp,
          type: 'api_call',
          apiMetadata: {
            model: trace.data.model,
            maxTokens: trace.data.max_tokens,
          },
        })
      } else if (trace.category === 'API_RESPONSE') {
        // Find the corresponding API call to add duration
        const lastApiCall = [...flow].reverse().find(f => f.type === 'api_call')
        if (lastApiCall && trace.data.duration_ms) {
          lastApiCall.apiMetadata!.duration = trace.data.duration_ms
        }

        flow.push({
          timestamp: trace.timestamp,
          type: 'response',
          responseMetadata: {
            id: trace.data.id,
            model: trace.data.model,
            stopReason: trace.data.stop_reason,
            inputTokens: trace.data.usage?.input_tokens || 0,
            outputTokens: trace.data.usage?.output_tokens || 0,
            duration: trace.data.duration_ms || 0,
          },
        })
      } else if (trace.category === 'ASSISTANT_RESPONSE') {
        flow.push({
          timestamp: trace.timestamp,
          type: 'response',
          assistantMessage: trace.data.message,
        })
      } else if (trace.category === 'TOOL_CALL') {
        flow.push({
          timestamp: trace.timestamp,
          type: 'tool_call',
          toolCall: {
            toolName: trace.data.toolName,
            toolInput: trace.data.toolInput,
            toolUseId: trace.data.toolUseId,
          },
        })
      } else if (trace.category === 'TOOL_RESULT') {
        flow.push({
          timestamp: trace.timestamp,
          type: 'tool_result',
          toolResult: {
            toolName: trace.data.toolName,
            result: trace.data.result,
            success: trace.data.success,
            error: trace.data.error,
          },
        })
      }
    }

    setConversationFlow(flow)
  }, [traces])

  useEffect(() => {
    if (autoScroll && tracesEndRef.current) {
      tracesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [conversationFlow, autoScroll])

  const formatText = (text: string) => {
    // Split by newlines and render as separate paragraphs
    return text.split('\n').map((line, idx) => (
      <span key={idx}>
        {line}
        {idx < text.split('\n').length - 1 && <br />}
      </span>
    ))
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    })
  }

  return (
    <div className="smart-traces-container">
      <div className="smart-traces-header">
        <h2>Smart View</h2>
        <p>Visual flow of agent activity with incremental changes</p>
      </div>

      <div className="smart-traces-controls">
        <label className="auto-scroll-toggle">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          Auto-scroll
        </label>
        <button className="clear-button" onClick={onClear}>
          Clear
        </button>
        <div className="flow-count">
          {conversationFlow.length} event{conversationFlow.length !== 1 ? 's' : ''}
        </div>
      </div>

      <div className="conversation-flow">
        {conversationFlow.length === 0 ? (
          <div className="empty-state">
            <p>No activity yet. Start a conversation in the Chat tab.</p>
          </div>
        ) : (
          conversationFlow.map((item, idx) => (
            <div key={idx} className={`flow-item flow-${item.type}`}>
              <div className="flow-timestamp">{formatTime(item.timestamp)}</div>

              {item.type === 'user' && item.userMessage && (
                <div className="flow-card user-message">
                  <div className="flow-card-header">
                    <span className="flow-icon">👤</span>
                    <span className="flow-label">User Message</span>
                  </div>
                  <div className="flow-content">
                    {formatText(item.userMessage)}
                  </div>
                </div>
              )}

              {item.type === 'api_call' && item.apiMetadata && (
                <div className="flow-card api-call">
                  <div className="flow-card-header">
                    <span className="flow-icon">🔄</span>
                    <span className="flow-label">API Request</span>
                  </div>
                  <div className="flow-metadata">
                    <div className="metadata-item">
                      <span className="metadata-label">Model:</span>
                      <span className="metadata-value">{item.apiMetadata.model}</span>
                    </div>
                    <div className="metadata-item">
                      <span className="metadata-label">Max Tokens:</span>
                      <span className="metadata-value">{item.apiMetadata.maxTokens}</span>
                    </div>
                    {item.apiMetadata.duration && (
                      <div className="metadata-item">
                        <span className="metadata-label">Duration:</span>
                        <span className="metadata-value">{item.apiMetadata.duration}ms</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {item.type === 'response' && (item.responseMetadata || item.assistantMessage) && (
                <div className="flow-card assistant-response">
                  {item.responseMetadata && (
                    <>
                      <div className="flow-card-header">
                        <span className="flow-icon">📊</span>
                        <span className="flow-label">API Response Metrics</span>
                      </div>
                      <div className="flow-metrics">
                        <div className="metric">
                          <div className="metric-value">{item.responseMetadata.inputTokens}</div>
                          <div className="metric-label">Input Tokens</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value">{item.responseMetadata.outputTokens}</div>
                          <div className="metric-label">Output Tokens</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value">{item.responseMetadata.duration}ms</div>
                          <div className="metric-label">Duration</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value">{item.responseMetadata.stopReason}</div>
                          <div className="metric-label">Stop Reason</div>
                        </div>
                      </div>
                    </>
                  )}

                  {item.assistantMessage && (
                    <>
                      <div className="flow-card-header">
                        <span className="flow-icon">🤖</span>
                        <span className="flow-label">Assistant Response</span>
                      </div>
                      <div className="flow-content">
                        {formatText(item.assistantMessage)}
                      </div>
                    </>
                  )}
                </div>
              )}

              {item.type === 'tool_call' && item.toolCall && (
                <div className="flow-card tool-call">
                  <div className="flow-card-header">
                    <span className="flow-icon">🔧</span>
                    <span className="flow-label">Tool Call: {item.toolCall.toolName}</span>
                  </div>
                  <div className="flow-content">
                    <div className="tool-info">
                      <div className="tool-info-row">
                        <span className="tool-info-label">Tool:</span>
                        <span className="tool-info-value">{item.toolCall.toolName}</span>
                      </div>
                      <div className="tool-info-row">
                        <span className="tool-info-label">Input:</span>
                      </div>
                      <pre className="tool-data">{JSON.stringify(item.toolCall.toolInput, null, 2)}</pre>
                    </div>
                  </div>
                </div>
              )}

              {item.type === 'tool_result' && item.toolResult && (
                <div className={`flow-card tool-result ${item.toolResult.success ? 'success' : 'error'}`}>
                  <div className="flow-card-header">
                    <span className="flow-icon">{item.toolResult.success ? '✅' : '❌'}</span>
                    <span className="flow-label">
                      Tool Result: {item.toolResult.toolName}
                      {item.toolResult.success ? ' (Success)' : ' (Error)'}
                    </span>
                  </div>
                  <div className="flow-content">
                    {item.toolResult.success ? (
                      <pre className="tool-data">{JSON.stringify(item.toolResult.result, null, 2)}</pre>
                    ) : (
                      <div className="tool-error">
                        <strong>Error:</strong> {item.toolResult.error}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
        <div ref={tracesEndRef} />
      </div>
    </div>
  )
}

export default SmartTraces
