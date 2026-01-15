import { useState } from 'react'
import './App.css'
import Chat from './Chat'
import Traces from './Traces'
import Settings from './Settings'

type View = 'chat' | 'traces' | 'settings'

function App() {
  const [currentView, setCurrentView] = useState<View>('chat')

  return (
    <div className="app">
      <header className="app-header">
        <h1>Claude Agent SDK Demo</h1>
        <p>A simple demonstration of Claude's Agent SDK capabilities</p>

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
        {currentView === 'chat' && <Chat />}
        {currentView === 'traces' && <Traces />}
        {currentView === 'settings' && <Settings />}
      </main>
    </div>
  )
}

export default App
