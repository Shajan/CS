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
  const [collapsedItems, setCollapsedItems] = useState<Set<number>>(new Set())
  const tracesEndRef = useRef<HTMLDivElement>(null)

  const toggleCollapse = (index: number) => {
    setCollapsedItems(prev => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        newSet.add(index)
      }
      return newSet
    })
  }

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
        // Find the corresponding API call and attach response metadata to it
        const lastApiCall = [...flow].reverse().find(f => f.type === 'api_call')
        if (lastApiCall) {
          lastApiCall.apiMetadata!.duration = trace.data.duration_ms
          lastApiCall.responseMetadata = {
            id: trace.data.id,
            model: trace.data.model,
            stopReason: trace.data.stop_reason,
            inputTokens: trace.data.usage?.input_tokens || 0,
            outputTokens: trace.data.usage?.output_tokens || 0,
            duration: trace.data.duration_ms || 0,
          }
        }
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

  const getRelativeTime = (timestamp: string): string => {
    const now = new Date()
    const date = new Date(timestamp)
    const diffMs = now.getTime() - date.getTime()
    const diffSec = Math.floor(diffMs / 1000)
    const diffMin = Math.floor(diffSec / 60)
    const diffHour = Math.floor(diffMin / 60)
    const diffDay = Math.floor(diffHour / 24)

    if (diffSec < 10) return 'just now'
    if (diffSec < 60) return `${diffSec}s ago`
    if (diffMin < 60) return `${diffMin}m ago`
    if (diffHour < 24) return `${diffHour}h ago`
    if (diffDay === 1) return 'yesterday'
    if (diffDay < 7) return `${diffDay}d ago`
    return date.toLocaleDateString()
  }

  const highlightJSON = (json: string): JSX.Element => {
    // Simple JSON syntax highlighting
    const parts = json.split(/("(?:[^"\\]|\\.)*")/g)

    return (
      <>
        {parts.map((part, i) => {
          if (i % 2 === 1) {
            // This is a quoted string
            if (part.match(/^"[^"]*":$/)) {
              // It's a key
              return <span key={i} className="json-key">{part}</span>
            } else {
              // It's a value
              return <span key={i} className="json-string">{part}</span>
            }
          } else {
            // Check for numbers, booleans, null
            return part.split(/\b/).map((word, j) => {
              if (word.match(/^-?\d+\.?\d*$/)) {
                return <span key={`${i}-${j}`} className="json-number">{word}</span>
              } else if (word === 'true' || word === 'false') {
                return <span key={`${i}-${j}`} className="json-boolean">{word}</span>
              } else if (word === 'null') {
                return <span key={`${i}-${j}`} className="json-null">{word}</span>
              }
              return word
            })
          }
        })}
      </>
    )
  }

  const calculateCost = (inputTokens: number, outputTokens: number, model: string): string => {
    // Pricing per million tokens (as of Jan 2025)
    const pricing: Record<string, { input: number; output: number }> = {
      'claude-sonnet-4-5-20250929': { input: 3.00, output: 15.00 },
      'claude-opus-4-5': { input: 15.00, output: 75.00 },
      'claude-3-5-sonnet': { input: 3.00, output: 15.00 },
      'claude-3-opus': { input: 15.00, output: 75.00 },
    }

    // Find matching pricing
    const modelKey = Object.keys(pricing).find(key => model.includes(key)) || 'claude-sonnet-4-5-20250929'
    const rates = pricing[modelKey]

    const inputCost = (inputTokens / 1_000_000) * rates.input
    const outputCost = (outputTokens / 1_000_000) * rates.output
    const totalCost = inputCost + outputCost

    // Format cost with appropriate precision
    if (totalCost < 0.0001) {
      return `<$0.0001`
    } else if (totalCost < 0.01) {
      return `$${totalCost.toFixed(4)}`
    } else {
      return `$${totalCost.toFixed(3)}`
    }
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
              <div className="flow-timestamp" title={formatTime(item.timestamp)}>
                {getRelativeTime(item.timestamp)}
              </div>

              {item.type === 'user' && item.userMessage && (
                <div className="flow-row user-message">
                  <div className="flow-label-col">USER</div>
                  <div className="flow-content-col">{formatText(item.userMessage)}</div>
                </div>
              )}

              {item.type === 'api_call' && item.apiMetadata && (
                <div className="flow-row api-call">
                  <div className="flow-label-col collapsible" onClick={() => toggleCollapse(idx)}>
                    API {collapsedItems.has(idx) ? '›' : '∨'}
                  </div>
                  <div className="flow-content-col">
                    <div className="api-summary">
                      {item.responseMetadata && (
                        <>
                          <span className="inline-metric">{item.responseMetadata.inputTokens}in</span>
                          <span className="inline-metric">{item.responseMetadata.outputTokens}out</span>
                          <span className="inline-metric">{item.responseMetadata.duration}ms</span>
                          <span className="inline-metric inline-cost">{calculateCost(item.responseMetadata.inputTokens, item.responseMetadata.outputTokens, item.responseMetadata.model)}</span>
                        </>
                      )}
                    </div>
                    {!collapsedItems.has(idx) && (
                      <div className="api-details">
                        model: {item.apiMetadata.model} | max_tokens: {item.apiMetadata.maxTokens}
                        {item.responseMetadata && ` | stop: ${item.responseMetadata.stopReason}`}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {item.type === 'response' && item.assistantMessage && (
                <div className="flow-row assistant-response">
                  <div className="flow-label-col">ASSISTANT</div>
                  <div className="flow-content-col">{formatText(item.assistantMessage)}</div>
                </div>
              )}

              {item.type === 'tool_call' && item.toolCall && (
                <div className="flow-row tool-call">
                  <div className="flow-label-col collapsible" onClick={() => toggleCollapse(idx)}>
                    {item.toolCall.toolName} {collapsedItems.has(idx) ? '›' : '∨'}
                  </div>
                  <div className="flow-content-col">
                    {!collapsedItems.has(idx) && (
                      <pre className="tool-data">{highlightJSON(JSON.stringify(item.toolCall.toolInput, null, 2))}</pre>
                    )}
                  </div>
                </div>
              )}

              {item.type === 'tool_result' && item.toolResult && (
                <div className={`flow-row tool-result ${item.toolResult.success ? 'success' : 'error'}`}>
                  <div className="flow-label-col collapsible" onClick={() => toggleCollapse(idx)}>
                    → {item.toolResult.success ? '' : 'ERR '}{collapsedItems.has(idx) ? '›' : '∨'}
                  </div>
                  <div className="flow-content-col">
                    {!collapsedItems.has(idx) && (
                      <>
                        {item.toolResult.success ? (
                          <pre className="tool-data">{highlightJSON(JSON.stringify(item.toolResult.result, null, 2))}</pre>
                        ) : (
                          <span className="tool-error">{item.toolResult.error}</span>
                        )}
                      </>
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
