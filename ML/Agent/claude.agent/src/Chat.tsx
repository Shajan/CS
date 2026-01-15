import { useState } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

const API_BASE_URL = 'http://localhost:3001/api/agent'

function Chat() {
  const [message, setMessage] = useState('')
  const [conversation, setConversation] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const sendMessage = async () => {
    if (!message.trim() || isLoading) return

    const userMessage = message.trim()
    setMessage('')
    setError(null)
    setIsLoading(true)

    // Add user message to conversation
    setConversation((prev) => [...prev, { role: 'user', content: userMessage }])

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response from agent')
      }

      const data = await response.json()

      // Add assistant response to conversation
      setConversation((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const clearConversation = async () => {
    try {
      await fetch(`${API_BASE_URL}/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      setConversation([])
      setError(null)
    } catch (err) {
      setError('Failed to clear conversation')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-view">
      <div className="agent-container">
        <div className="header-row">
          <h2>Agent Interaction</h2>
          {conversation.length > 0 && (
            <button className="clear-button" onClick={clearConversation}>
              Clear Chat
            </button>
          )}
        </div>

        <div className="conversation-area">
          {conversation.length === 0 ? (
            <p className="info-text">
              Start a conversation with the agent by typing a message below.
              The agent is powered by Claude and runs securely on the server.
            </p>
          ) : (
            <div className="messages">
              {conversation.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <strong>{msg.role === 'user' ? 'You' : 'Agent'}:</strong>
                  <p>{msg.content}</p>
                </div>
              ))}
              {isLoading && (
                <div className="message assistant loading">
                  <strong>Agent:</strong>
                  <p>Thinking...</p>
                </div>
              )}
            </div>
          )}
        </div>

        {error && <div className="error-message">{error}</div>}

        <div className="chat-interface">
          <textarea
            className="message-input"
            placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            rows={3}
            disabled={isLoading}
          />
          <button
            className="send-button"
            onClick={sendMessage}
            disabled={isLoading || !message.trim()}
          >
            {isLoading ? 'Sending...' : 'Send Message'}
          </button>
        </div>
      </div>

      <aside className="info-panel">
        <h3>Getting Started</h3>
        <ol>
          <li>Add your Anthropic API key to <code>.env</code></li>
          <li>Install dependencies with <code>pnpm install</code></li>
          <li>Start the dev server with <code>pnpm dev</code></li>
          <li>Chat with the agent!</li>
        </ol>

        <h3>Architecture</h3>
        <ul>
          <li>
            <strong>Frontend:</strong> React + TypeScript
          </li>
          <li>
            <strong>Backend:</strong> Express server
          </li>
          <li>
            <strong>Agent:</strong> Runs on server (API key secure)
          </li>
        </ul>

        <h3>Agent Features</h3>
        <ul>
          <li>Conversational memory</li>
          <li>Context awareness</li>
          <li>Easy to extend</li>
        </ul>
      </aside>
    </div>
  )
}

export default Chat
