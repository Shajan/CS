import { useState, useEffect } from 'react'
import './Settings.css'

interface Tool {
  name: string
  description: string
  enabled: boolean
}

interface MCPServer {
  id: string
  name: string
  description: string
  enabled: boolean
  status: 'connected' | 'disconnected' | 'error'
}

const API_BASE_URL = 'http://localhost:3001/api'

function Settings() {
  const [activeTab, setActiveTab] = useState<'tools' | 'mcp'>('tools')
  const [tools, setTools] = useState<Tool[]>([])
  const [mcpServers, setMCPServers] = useState<MCPServer[]>([])
  const [showAddMCP, setShowAddMCP] = useState(false)
  const [newMCPUrl, setNewMCPUrl] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadTools()
    loadMCPServers()
  }, [])

  const loadTools = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tools`)
      const data = await response.json()
      setTools(data.tools || [])
    } catch (error) {
      console.error('Failed to load tools:', error)
    }
  }

  const loadMCPServers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/mcp/servers`)
      const data = await response.json()
      setMCPServers(data.servers || [])
    } catch (error) {
      console.error('Failed to load MCP servers:', error)
    }
  }

  const toggleTool = async (toolName: string) => {
    try {
      const tool = tools.find(t => t.name === toolName)
      if (!tool) return

      await fetch(`${API_BASE_URL}/tools/${toolName}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !tool.enabled })
      })

      setTools(tools.map(t =>
        t.name === toolName ? { ...t, enabled: !t.enabled } : t
      ))
    } catch (error) {
      console.error('Failed to toggle tool:', error)
    }
  }

  const toggleMCPServer = async (serverId: string) => {
    try {
      const server = mcpServers.find(s => s.id === serverId)
      if (!server) return

      await fetch(`${API_BASE_URL}/mcp/servers/${serverId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !server.enabled })
      })

      setMCPServers(mcpServers.map(s =>
        s.id === serverId ? { ...s, enabled: !s.enabled } : s
      ))
    } catch (error) {
      console.error('Failed to toggle MCP server:', error)
    }
  }

  const addMCPServer = async () => {
    if (!newMCPUrl.trim()) return

    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/mcp/servers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: newMCPUrl })
      })

      if (response.ok) {
        setNewMCPUrl('')
        setShowAddMCP(false)
        await loadMCPServers()
      }
    } catch (error) {
      console.error('Failed to add MCP server:', error)
    } finally {
      setLoading(false)
    }
  }

  const removeMCPServer = async (serverId: string) => {
    if (!confirm('Remove this MCP server?')) return

    try {
      await fetch(`${API_BASE_URL}/mcp/servers/${serverId}`, {
        method: 'DELETE'
      })

      setMCPServers(mcpServers.filter(s => s.id !== serverId))
    } catch (error) {
      console.error('Failed to remove MCP server:', error)
    }
  }

  return (
    <div className="settings-container">
      <div className="settings-header">
        <h1>Settings</h1>
        <p>Configure tools and MCP servers</p>
      </div>

      <div className="settings-tabs">
        <button
          className={`tab-button ${activeTab === 'tools' ? 'active' : ''}`}
          onClick={() => setActiveTab('tools')}
        >
          Tools
        </button>
        <button
          className={`tab-button ${activeTab === 'mcp' ? 'active' : ''}`}
          onClick={() => setActiveTab('mcp')}
        >
          MCP Servers
        </button>
      </div>

      <div className="settings-content">
        {activeTab === 'tools' && (
          <div className="tools-section">
            <div className="section-header">
              <h2>Available Tools</h2>
              <span className="item-count">{tools.filter(t => t.enabled).length}/{tools.length} enabled</span>
            </div>

            {tools.length === 0 ? (
              <div className="empty-state">No tools available</div>
            ) : (
              <div className="items-list">
                {tools.map(tool => (
                  <div key={tool.name} className="item-row">
                    <div className="item-info">
                      <div className="item-name">{tool.name}</div>
                      <div className="item-description">{tool.description}</div>
                    </div>
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={tool.enabled}
                        onChange={() => toggleTool(tool.name)}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'mcp' && (
          <div className="mcp-section">
            <div className="section-header">
              <h2>MCP Servers</h2>
              <button className="add-button" onClick={() => setShowAddMCP(!showAddMCP)}>
                {showAddMCP ? 'Cancel' : '+ Add Server'}
              </button>
            </div>

            {showAddMCP && (
              <div className="add-mcp-form">
                <input
                  type="text"
                  className="mcp-url-input"
                  placeholder="Enter MCP server URL or command..."
                  value={newMCPUrl}
                  onChange={(e) => setNewMCPUrl(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && addMCPServer()}
                />
                <button
                  className="add-confirm-button"
                  onClick={addMCPServer}
                  disabled={loading || !newMCPUrl.trim()}
                >
                  {loading ? 'Adding...' : 'Add'}
                </button>
              </div>
            )}

            {mcpServers.length === 0 ? (
              <div className="empty-state">
                No MCP servers installed. Click "Add Server" to get started.
              </div>
            ) : (
              <div className="items-list">
                {mcpServers.map(server => (
                  <div key={server.id} className="item-row mcp-row">
                    <div className="item-info">
                      <div className="item-name-row">
                        <span className="item-name">{server.name}</span>
                        <span className={`status-badge status-${server.status}`}>
                          {server.status}
                        </span>
                      </div>
                      <div className="item-description">{server.description}</div>
                    </div>
                    <div className="item-actions">
                      <label className="toggle-switch">
                        <input
                          type="checkbox"
                          checked={server.enabled}
                          onChange={() => toggleMCPServer(server.id)}
                        />
                        <span className="toggle-slider"></span>
                      </label>
                      <button
                        className="remove-button"
                        onClick={() => removeMCPServer(server.id)}
                        title="Remove server"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Settings
