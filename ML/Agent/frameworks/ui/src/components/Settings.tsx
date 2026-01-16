import { useState, useEffect } from 'react';
import {
  listFrameworks,
  getFrameworkSettings,
  updateFrameworkSettings,
  getMCPServers,
  addMCPServer,
  updateMCPServer,
  deleteMCPServer,
  type FrameworkInfo,
} from '../services/api';
import './Settings.css';

interface MCPServer {
  name: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  enabled: boolean;
}

export default function Settings() {
  const [frameworks, setFrameworks] = useState<FrameworkInfo[]>([]);
  const [mcpServers, setMCPServers] = useState<MCPServer[]>([]);
  const [selectedFramework, setSelectedFramework] = useState<string>('');
  const [frameworkSettings, setFrameworkSettings] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'frameworks' | 'mcp'>('frameworks');

  // New MCP server form
  const [newMCPName, setNewMCPName] = useState('');
  const [newMCPCommand, setNewMCPCommand] = useState('');
  const [newMCPArgs, setNewMCPArgs] = useState('');
  const [showAddMCP, setShowAddMCP] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [fwData, mcpData] = await Promise.all([
        listFrameworks(),
        getMCPServers(),
      ]);
      setFrameworks(fwData);
      setMCPServers(mcpData);
      if (fwData.length > 0 && !selectedFramework) {
        setSelectedFramework(fwData[0].name);
        await loadFrameworkSettings(fwData[0].name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const loadFrameworkSettings = async (framework: string) => {
    try {
      const data = await getFrameworkSettings(framework);
      setFrameworkSettings(data.settings || {});
    } catch (err) {
      console.error('Failed to load framework settings:', err);
    }
  };

  const handleFrameworkChange = async (framework: string) => {
    setSelectedFramework(framework);
    await loadFrameworkSettings(framework);
  };

  const handleSaveFrameworkSettings = async () => {
    try {
      await updateFrameworkSettings(selectedFramework, frameworkSettings);
      alert('Settings saved successfully');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to save settings');
    }
  };

  const handleAddMCP = async () => {
    if (!newMCPName || !newMCPCommand) {
      alert('Name and command are required');
      return;
    }

    try {
      const server: MCPServer = {
        name: newMCPName,
        command: newMCPCommand,
        args: newMCPArgs ? newMCPArgs.split(' ') : [],
        enabled: true,
      };
      await addMCPServer(server);
      await loadData();
      setNewMCPName('');
      setNewMCPCommand('');
      setNewMCPArgs('');
      setShowAddMCP(false);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to add MCP server');
    }
  };

  const handleToggleMCP = async (name: string, enabled: boolean) => {
    try {
      await updateMCPServer(name, { enabled });
      await loadData();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to update MCP server');
    }
  };

  const handleDeleteMCP = async (name: string) => {
    if (!confirm(`Delete MCP server "${name}"?`)) return;

    try {
      await deleteMCPServer(name);
      await loadData();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete MCP server');
    }
  };

  const handleToolToggle = (tool: string, enabled: boolean) => {
    const enabledTools = frameworkSettings.enabledTools || [];
    const newTools = enabled
      ? [...enabledTools, tool]
      : enabledTools.filter((t: string) => t !== tool);
    setFrameworkSettings({ ...frameworkSettings, enabledTools: newTools });
  };

  const selectedFw = frameworks.find(f => f.name === selectedFramework);
  const availableTools = ['calculator', 'web-search', 'file-system', 'code-execution'];
  const enabledTools = frameworkSettings.enabledTools || [];

  if (loading) {
    return <div className="settings-container"><p>Loading...</p></div>;
  }

  return (
    <div className="settings-container">
      <div className="settings-header">
        <h1>Settings</h1>
        <p>Configure frameworks and MCP servers</p>
      </div>

      <div className="settings-tabs">
        <button
          className={`tab ${activeTab === 'frameworks' ? 'active' : ''}`}
          onClick={() => setActiveTab('frameworks')}
        >
          Frameworks
        </button>
        <button
          className={`tab ${activeTab === 'mcp' ? 'active' : ''}`}
          onClick={() => setActiveTab('mcp')}
        >
          MCP Servers
        </button>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {activeTab === 'frameworks' && (
        <div className="frameworks-section">
          <div className="framework-selector-section">
            <label htmlFor="framework-select">Framework:</label>
            <select
              id="framework-select"
              value={selectedFramework}
              onChange={(e) => handleFrameworkChange(e.target.value)}
            >
              {frameworks.map((fw) => (
                <option key={fw.name} value={fw.name}>
                  {fw.displayName}
                </option>
              ))}
            </select>
          </div>

          {selectedFw && (
            <div className="framework-config">
              <div className="config-section">
                <h3>Enabled Tools</h3>
                <p className="section-description">
                  Select which tools are available for this framework
                </p>
                <div className="tools-list">
                  {availableTools.map((tool) => (
                    <label key={tool} className="tool-item">
                      <input
                        type="checkbox"
                        checked={enabledTools.includes(tool)}
                        onChange={(e) => handleToolToggle(tool, e.target.checked)}
                      />
                      <span>{tool}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="config-actions">
                <button onClick={handleSaveFrameworkSettings} className="save-button">
                  Save Settings
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'mcp' && (
        <div className="mcp-section">
          <div className="mcp-header">
            <h3>MCP Servers ({mcpServers.length})</h3>
            <button onClick={() => setShowAddMCP(!showAddMCP)} className="add-button">
              {showAddMCP ? 'Cancel' : 'Add Server'}
            </button>
          </div>

          {showAddMCP && (
            <div className="add-mcp-form">
              <div className="form-group">
                <label>Name:</label>
                <input
                  type="text"
                  value={newMCPName}
                  onChange={(e) => setNewMCPName(e.target.value)}
                  placeholder="e.g., filesystem"
                />
              </div>
              <div className="form-group">
                <label>Command:</label>
                <input
                  type="text"
                  value={newMCPCommand}
                  onChange={(e) => setNewMCPCommand(e.target.value)}
                  placeholder="e.g., npx -y @modelcontextprotocol/server-filesystem"
                />
              </div>
              <div className="form-group">
                <label>Arguments (optional):</label>
                <input
                  type="text"
                  value={newMCPArgs}
                  onChange={(e) => setNewMCPArgs(e.target.value)}
                  placeholder="e.g., /path/to/dir"
                />
              </div>
              <button onClick={handleAddMCP} className="save-button">
                Add Server
              </button>
            </div>
          )}

          <div className="mcp-servers-list">
            {mcpServers.length === 0 ? (
              <p className="empty-state">No MCP servers configured</p>
            ) : (
              mcpServers.map((server) => (
                <div key={server.name} className="mcp-server-item">
                  <div className="server-info">
                    <div className="server-name">
                      <strong>{server.name}</strong>
                      <span className={`status ${server.enabled ? 'enabled' : 'disabled'}`}>
                        {server.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                    <div className="server-command">
                      <code>{server.command}</code>
                      {server.args && server.args.length > 0 && (
                        <code className="args">{server.args.join(' ')}</code>
                      )}
                    </div>
                  </div>
                  <div className="server-actions">
                    <button
                      onClick={() => handleToggleMCP(server.name, !server.enabled)}
                      className="toggle-button"
                    >
                      {server.enabled ? 'Disable' : 'Enable'}
                    </button>
                    <button
                      onClick={() => handleDeleteMCP(server.name)}
                      className="delete-button"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
