import { useState, useEffect, useRef } from 'react';
import { sendChatMessage, clearHistory, type Message, type ChatResponse } from '../services/api';
import { FrameworkSelector } from './FrameworkSelector';
import './Chat.css';

interface ChatProps {
  onViewTraces?: (sessionId: string, framework: string) => void;
}

export function Chat({ onViewTraces }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => `session-${Date.now()}`);
  const [framework, setFramework] = useState('claude-agent');
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await sendChatMessage({
        message: input,
        sessionId,
        framework,
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.response,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setLastResponse(response);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (confirm('Clear conversation history?')) {
      try {
        await clearHistory(sessionId, framework);
        setMessages([]);
        setLastResponse(null);
      } catch (error) {
        console.error('Error clearing history:', error);
      }
    }
  };

  const handleFrameworkChange = (newFramework: string) => {
    if (messages.length > 0) {
      if (
        !confirm(
          'Switching frameworks will clear the current conversation. Continue?'
        )
      ) {
        return;
      }
      setMessages([]);
      setLastResponse(null);
    }
    setFramework(newFramework);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <FrameworkSelector
          selectedFramework={framework}
          onFrameworkChange={handleFrameworkChange}
        />
      </div>

      {messages.length > 0 && onViewTraces && (
        <div className="traces-action">
          <button
            className="view-traces-button"
            onClick={() => onViewTraces(sessionId, framework)}
          >
            View Traces
          </button>
        </div>
      )}

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <p>Type a message to start</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-role">{message.role === 'user' ? 'You' : 'Assistant'}</div>
            <div className="message-content">{message.content}</div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="message-role">Assistant</div>
            <div className="message-content typing">Thinking...</div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {lastResponse?.metadata && (
        <div className="response-metadata">
          <span>Framework: {lastResponse.framework}</span>
          {lastResponse.metadata.model && <span>Model: {lastResponse.metadata.model}</span>}
          {lastResponse.metadata.duration && (
            <span>Duration: {lastResponse.metadata.duration}ms</span>
          )}
          {lastResponse.metadata.tokensUsed && (
            <span>Tokens: {lastResponse.metadata.tokensUsed}</span>
          )}
        </div>
      )}

      <div className="input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={loading}
          rows={3}
        />
        <div className="button-group">
          <button onClick={handleSend} disabled={loading || !input.trim()}>
            {loading ? 'Sending...' : 'Send'}
          </button>
          <button onClick={handleClear} disabled={loading || messages.length === 0}>
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
