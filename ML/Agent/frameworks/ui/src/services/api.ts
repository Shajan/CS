/**
 * API service for communicating with the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

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

export interface ConversationHistory {
  sessionId: string;
  messages: Message[];
  framework?: string;
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
    const error = await response.json();
    throw new Error(error.message || 'Failed to send message');
  }

  return response.json();
}

/**
 * Get conversation history
 */
export async function getHistory(
  sessionId: string,
  framework?: string
): Promise<ConversationHistory> {
  const params = new URLSearchParams({ sessionId });
  if (framework) params.append('framework', framework);

  const response = await fetch(`${API_BASE_URL}/api/history?${params}`);

  if (!response.ok) {
    throw new Error('Failed to get history');
  }

  return response.json();
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
 * List all available frameworks
 */
export async function listFrameworks(): Promise<FrameworkInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/frameworks`);

  if (!response.ok) {
    throw new Error('Failed to list frameworks');
  }

  const data = await response.json();
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

  return response.json();
}

/**
 * Get trace events for a session
 */
export async function getTraces(sessionId: string): Promise<any[]> {
  const params = new URLSearchParams({ sessionId });

  const response = await fetch(`${API_BASE_URL}/api/traces?${params}`);

  if (!response.ok) {
    throw new Error('Failed to get traces');
  }

  const data = await response.json();
  return data.traces;
}

/**
 * Get all settings
 */
export async function getSettings(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/settings`);

  if (!response.ok) {
    throw new Error('Failed to get settings');
  }

  return response.json();
}

/**
 * Get framework settings
 */
export async function getFrameworkSettings(framework: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/settings/framework/${framework}`);

  if (!response.ok) {
    throw new Error(`Failed to get settings for ${framework}`);
  }

  return response.json();
}

/**
 * Update framework settings
 */
export async function updateFrameworkSettings(framework: string, settings: any): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/settings/framework/${framework}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });

  if (!response.ok) {
    throw new Error(`Failed to update settings for ${framework}`);
  }

  return response.json();
}

/**
 * Get MCP servers
 */
export async function getMCPServers(): Promise<any[]> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp`);

  if (!response.ok) {
    throw new Error('Failed to get MCP servers');
  }

  const data = await response.json();
  return data.servers;
}

/**
 * Add MCP server
 */
export async function addMCPServer(server: any): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(server),
  });

  if (!response.ok) {
    throw new Error('Failed to add MCP server');
  }

  return response.json();
}

/**
 * Update MCP server
 */
export async function updateMCPServer(name: string, updates: any): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp/${name}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    throw new Error(`Failed to update MCP server ${name}`);
  }

  return response.json();
}

/**
 * Delete MCP server
 */
export async function deleteMCPServer(name: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/settings/mcp/${name}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete MCP server ${name}`);
  }

  return response.json();
}
