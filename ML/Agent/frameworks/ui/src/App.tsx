import { useState } from 'react';
import { Chat } from './components/Chat';
import Traces from './components/Traces';
import Settings from './components/Settings';
import './App.css';

type View = 'chat' | 'traces' | 'settings';

function App() {
  const [currentView, setCurrentView] = useState<View>('chat');
  const [traceSessionId, setTraceSessionId] = useState<string>('');
  const [traceFramework, setTraceFramework] = useState<string>('');

  const handleViewTraces = (sessionId: string, framework: string) => {
    setTraceSessionId(sessionId);
    setTraceFramework(framework);
    setCurrentView('traces');
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Multi-Framework Agent System</h1>
        <p>Compare and test different AI agent frameworks</p>

        <nav className="app-nav">
          <button
            className={`nav-button ${currentView === 'chat' ? 'active' : ''}`}
            onClick={() => setCurrentView('chat')}
          >
            Chat
          </button>
          <button
            className={`nav-button ${currentView === 'settings' ? 'active' : ''}`}
            onClick={() => setCurrentView('settings')}
          >
            Settings
          </button>
          <button
            className={`nav-button ${currentView === 'traces' ? 'active' : ''}`}
            onClick={() => setCurrentView('traces')}
          >
            Traces
          </button>
        </nav>
      </header>

      <main className="app-main">
        {currentView === 'chat' && <Chat onViewTraces={handleViewTraces} />}
        {currentView === 'traces' && <Traces initialSessionId={traceSessionId} initialFramework={traceFramework} />}
        {currentView === 'settings' && <Settings />}
      </main>
    </div>
  );
}

export default App;
