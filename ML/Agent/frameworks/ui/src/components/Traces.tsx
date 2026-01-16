import { useState, useEffect } from 'react';
import { getTraces } from '../services/api';
import SmartTraces from './SmartTraces';
import './Traces.css';

interface TracesProps {
  initialSessionId?: string;
  initialFramework?: string; // Kept for compatibility but not used
}

type ViewMode = 'smart' | 'raw';

export default function Traces({ initialSessionId = '' }: TracesProps) {
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [traces, setTraces] = useState<any[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('smart');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHistoryForSession = async (sid: string) => {
    if (!sid.trim()) {
      setError('Please enter a session ID');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const tracesData = await getTraces(sid);
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

  const handleCopyJSON = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(traces, null, 2));
      // Could add a toast notification here
    } catch (err) {
      console.error('Failed to copy JSON:', err);
    }
  };

  const handleDownloadJSON = () => {
    const jsonStr = JSON.stringify(traces, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `traces-${sessionId}-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Auto-load history if session ID is provided
  useEffect(() => {
    if (initialSessionId) {
      setSessionId(initialSessionId);
      loadHistoryForSession(initialSessionId);
    }
  }, [initialSessionId]);

  const loadHistory = async () => {
    loadHistoryForSession(sessionId);
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

        <button onClick={loadHistory} disabled={loading || !sessionId.trim()}>
          {loading ? 'Loading...' : 'Load Traces'}
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
            <div className="raw-logs-actions">
              <button onClick={handleCopyJSON} className="action-button">Copy JSON</button>
              <button onClick={handleDownloadJSON} className="action-button">Download JSON</button>
              <button onClick={handleClearTraces} className="clear-button">Clear</button>
            </div>
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
            <strong>Tip:</strong> Session IDs are generated automatically when you start a conversation.
            Check your browser's network tab or console to find them. They look like: <code>session-1737073891234</code>
          </div>
        </div>
      )}
    </div>
  );
}
