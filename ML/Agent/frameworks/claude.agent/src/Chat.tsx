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
          <h2>Chat</h2>
          {conversation.length > 0 && (
            <button className="clear-button" onClick={clearConversation}>
              Clear
            </button>
          )}
        </div>

        <div className="conversation-area">
          {conversation.length === 0 ? (
            <p className="info-text">
              No messages yet
            </p>
          ) : (
            <div className="messages">
              {conversation.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <span className="message-role">{msg.role === 'user' ? 'USER' : 'ASSISTANT'}</span>
                  <span className="message-content">{msg.content}</span>
                </div>
              ))}
              {isLoading && (
                <div className="message assistant loading">
                  <span className="message-role">ASSISTANT</span>
                  <span className="message-content">Thinking...</span>
                </div>
              )}
            </div>
          )}
        </div>

        {error && <div className="error-message">{error}</div>}

        <div className="chat-interface">
          <textarea
            className="message-input"
            placeholder="Type message... (Enter to send, Shift+Enter for new line)"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            rows={2}
            disabled={isLoading}
          />
          <button
            className="send-button"
            onClick={sendMessage}
            disabled={isLoading || !message.trim()}
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default Chat
