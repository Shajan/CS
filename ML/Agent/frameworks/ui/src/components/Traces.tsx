import { useState, useEffect } from 'react';
import { getTraces } from '../services/api';
import SmartTraces from './SmartTraces';
import './Traces.css';

interface TracesProps {
  initialSessionId?: string;
  initialFramework?: string;
}

type ViewMode = 'smart' | 'raw';

export default function Traces({ initialSessionId = '', initialFramework = 'claude-agent' }: TracesProps) {
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [framework, setFramework] = useState(initialFramework);
  const [traces, setTraces] = useState<any[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('smart');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHistoryForSession = async (sid: string, fw: string) => {
    if (!sid.trim()) {
      setError('Please enter a session ID');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const tracesData = await getTraces(sid, fw);
      setTraces(tracesData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load traces');
      setTraces([]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearTraces = () => {
    setTraces([]);
  };

  // Auto-load history if session ID is provided
  useEffect(() => {
    if (initialSessionId && initialFramework) {
      setSessionId(initialSessionId);
      setFramework(initialFramework);
      loadHistoryForSession(initialSessionId, initialFramework);
    }
  }, [initialSessionId, initialFramework]);

  const loadHistory = async () => {
    loadHistoryForSession(sessionId, framework);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      loadHistory();
    }
  };

  return (
    <div className="traces-container">
      <div className="history-controls">
        <div className="input-group">
          <label htmlFor="session-id">Session ID:</label>
          <input
            id="session-id"
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="e.g., session-1234567890"
          />
        </div>

        <div className="input-group">
          <label htmlFor="framework-select">Framework:</label>
          <select
            id="framework-select"
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
          >
            <option value="claude-agent">Claude Agent SDK</option>
            <option value="langchain">LangChain</option>
          </select>
        </div>

        <button onClick={loadHistory} disabled={loading || !sessionId.trim()}>
          {loading ? 'Loading...' : 'Load History'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {traces.length > 0 && (
        <div className="view-mode-tabs">
          <button
            className={`tab-button ${viewMode === 'smart' ? 'active' : ''}`}
            onClick={() => setViewMode('smart')}
          >
            Smart View
          </button>
          <button
            className={`tab-button ${viewMode === 'raw' ? 'active' : ''}`}
            onClick={() => setViewMode('raw')}
          >
            Raw Logs
          </button>
        </div>
      )}

      {viewMode === 'smart' && traces.length > 0 && (
        <SmartTraces traces={traces} onClear={handleClearTraces} />
      )}

      {viewMode === 'raw' && traces.length > 0 && (
        <div className="raw-logs">
          <div className="raw-logs-header">
            <h3>Raw Trace Events ({traces.length})</h3>
            <button onClick={handleClearTraces} className="clear-button">Clear</button>
          </div>
          <pre className="raw-logs-content">
            {JSON.stringify(traces, null, 2)}
          </pre>
        </div>
      )}

      {traces.length === 0 && !loading && !error && (
        <div className="empty-state">
          <p>Enter a session ID above to view conversation data</p>
          <div className="example-hint">
            <strong>Tip:</strong> Session IDs are shown in the Chat tab when you send messages.
            They look like: <code>session-1737073891234</code>
          </div>
        </div>
      )}
    </div>
  );
}
