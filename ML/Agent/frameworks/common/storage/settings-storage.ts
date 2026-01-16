/**
 * Settings storage for framework configurations and MCP servers
 */

export interface FrameworkSettings {
  [frameworkName: string]: {
    enabledTools?: string[];
    customConfig?: Record<string, any>;
  };
}

export interface MCPServer {
  name: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  enabled: boolean;
}

export interface Settings {
  frameworks: FrameworkSettings;
  mcpServers: MCPServer[];
}

class SettingsStorage {
  private settings: Settings = {
    frameworks: {},
    mcpServers: [],
  };

  getSettings(): Settings {
    return { ...this.settings };
  }

  getFrameworkSettings(frameworkName: string): Record<string, any> {
    return { ...this.settings.frameworks[frameworkName] } || {};
  }

  setFrameworkSettings(frameworkName: string, config: Record<string, any>): void {
    this.settings.frameworks[frameworkName] = config;
  }

  getMCPServers(): MCPServer[] {
    return [...this.settings.mcpServers];
  }

  addMCPServer(server: MCPServer): void {
    this.settings.mcpServers.push(server);
  }

  updateMCPServer(name: string, updates: Partial<MCPServer>): void {
    const index = this.settings.mcpServers.findIndex(s => s.name === name);
    if (index !== -1) {
      this.settings.mcpServers[index] = {
        ...this.settings.mcpServers[index],
        ...updates,
      };
    }
  }

  removeMCPServer(name: string): void {
    this.settings.mcpServers = this.settings.mcpServers.filter(s => s.name !== name);
  }

  reset(): void {
    this.settings = {
      frameworks: {},
      mcpServers: [],
    };
  }
}

export const settingsStorage = new SettingsStorage();
