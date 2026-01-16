/**
 * API client for communicating with the backend server
 */

const API_BASE_URL = process.env.API_URL || 'http://localhost:3001';

export interface ChatRequest {
  message: string;
  sessionId: string;
  framework?: string;
  config?: Record<string, any>;
}

export interface ChatResponse {
  response: string;
  sessionId: string;
  framework: string;
  metadata?: {
    model?: string;
    tokensUsed?: number;
    duration?: number;
    [key: string]: any;
  };
}

export interface FrameworkInfo {
  name: string;
  displayName: string;
  version: string;
  description?: string;
  capabilities?: {
    supportsStreaming?: boolean;
    supportsTools?: boolean;
    supportsMultiModal?: boolean;
    supportsMultiAgent?: boolean;
    supportsMemory?: boolean;
    maxContextLength?: number;
    supportedModels?: string[];
  };
}

/**
 * Send a chat message
 */
export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json() as any;
    throw new Error(error.message || error.error || 'Failed to send message');
  }

  return response.json() as Promise<ChatResponse>;
}

/**
 * List all available frameworks
 */
export async function listFrameworks(): Promise<FrameworkInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/frameworks`);

  if (!response.ok) {
    throw new Error('Failed to list frameworks');
  }

  const data = await response.json() as { frameworks: FrameworkInfo[] };
  return data.frameworks;
}

/**
 * Get details about a specific framework
 */
export async function getFrameworkInfo(name: string): Promise<FrameworkInfo> {
  const response = await fetch(`${API_BASE_URL}/api/frameworks/${name}`);

  if (!response.ok) {
    throw new Error(`Failed to get framework info for ${name}`);
  }

  return response.json() as Promise<FrameworkInfo>;
}

/**
 * Clear conversation history
 */
export async function clearHistory(
  sessionId: string,
  framework?: string
): Promise<void> {
  const params = new URLSearchParams({ sessionId });
  if (framework) params.append('framework', framework);

  const response = await fetch(`${API_BASE_URL}/api/history?${params}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Failed to clear history');
  }
}

/**
 * MCP Server interface
 */
export interface MCPServer {
  name: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  enabled: boolean;
}

/**
 * Get settings for a specific framework
 */
export async function getFrameworkSettings(framework: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/settings/framework/${framework}`);

  if (!response.ok) {
    throw new Error(`Failed to get settings for ${framework}`);
  }

  const data = await response.json() as { framework: string; settings: any };
  return data.settings;
}

/**
 * Update settings for a specific framework
 */
export async function updateFrameworkSettings(framework: string, settings: any): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/settings/framework/${framework}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });

  if (!response.ok) {
    throw new Error(`Failed to update settings for ${framework}`);
  }
}

/**
 * Get all MCP servers
 */
export async function getMCPServers(): Promise<MCPServer[]> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp`);

  if (!response.ok) {
    throw new Error('Failed to get MCP servers');
  }

  const data = await response.json() as { servers: MCPServer[] };
  return data.servers;
}

/**
 * Add a new MCP server
 */
export async function addMCPServer(server: MCPServer): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(server),
  });

  if (!response.ok) {
    const error = await response.json() as any;
    throw new Error(error.message || error.error || 'Failed to add MCP server');
  }
}

/**
 * Update an existing MCP server
 */
export async function updateMCPServer(name: string, updates: Partial<MCPServer>): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp/${name}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    throw new Error(`Failed to update MCP server ${name}`);
  }
}

/**
 * Delete an MCP server
 */
export async function deleteMCPServer(name: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp/${name}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete MCP server ${name}`);
  }
}
