import { Router, Request, Response } from 'express';
import { randomUUID } from 'crypto';

const router = Router();

interface MCPServer {
  id: string;
  name: string;
  url: string;
  description: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'error';
}

// Store MCP servers (in production, use database)
const mcpServers = new Map<string, MCPServer>();

/**
 * GET /api/mcp/servers
 * Get all MCP servers
 */
router.get('/servers', (req: Request, res: Response) => {
  const servers = Array.from(mcpServers.values());
  res.json({ servers });
});

/**
 * POST /api/mcp/servers
 * Add a new MCP server
 */
router.post('/servers', async (req: Request, res: Response) => {
  const { url, name, description } = req.body;

  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }

  const serverId = randomUUID();

  // Extract name from URL if not provided
  const serverName = name || extractNameFromUrl(url);
  const serverDescription = description || `MCP server at ${url}`;

  const server: MCPServer = {
    id: serverId,
    name: serverName,
    url,
    description: serverDescription,
    enabled: true,
    status: 'disconnected', // Will be updated when connection is established
  };

  mcpServers.set(serverId, server);

  // Try to connect to the server
  try {
    // TODO: Implement actual MCP connection logic
    // For now, just set status to connected
    server.status = 'connected';
  } catch (error) {
    server.status = 'error';
  }

  res.json({
    success: true,
    server,
  });
});

/**
 * PATCH /api/mcp/servers/:id
 * Enable or disable an MCP server
 */
router.patch('/servers/:id', (req: Request, res: Response) => {
  const { id } = req.params;
  const { enabled } = req.body;

  const server = mcpServers.get(id);
  if (!server) {
    return res.status(404).json({ error: 'MCP server not found' });
  }

  if (typeof enabled !== 'boolean') {
    return res.status(400).json({ error: 'enabled must be a boolean' });
  }

  server.enabled = enabled;

  // Update status based on enabled state
  if (!enabled) {
    server.status = 'disconnected';
  } else {
    // Try to reconnect
    server.status = 'connected'; // TODO: Implement actual connection logic
  }

  res.json({
    success: true,
    server,
  });
});

/**
 * DELETE /api/mcp/servers/:id
 * Remove an MCP server
 */
router.delete('/servers/:id', (req: Request, res: Response) => {
  const { id } = req.params;

  const server = mcpServers.get(id);
  if (!server) {
    return res.status(404).json({ error: 'MCP server not found' });
  }

  // TODO: Disconnect from server before removing
  mcpServers.delete(id);

  res.json({
    success: true,
    message: 'MCP server removed',
  });
});

/**
 * Helper function to extract a friendly name from URL
 */
function extractNameFromUrl(url: string): string {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname || url;
  } catch {
    // If not a valid URL, assume it's a command
    const parts = url.split(/[\s/]+/);
    return parts[parts.length - 1] || url;
  }
}

export default router;
