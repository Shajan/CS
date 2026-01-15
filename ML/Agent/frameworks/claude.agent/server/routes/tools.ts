import { Router, Request, Response } from 'express';
import { tools } from '../tools/index.js';

const router = Router();

// Store tool enabled state (in production, use database)
const toolState = new Map<string, boolean>();

// Initialize all tools as enabled by default
Object.keys(tools).forEach(toolName => {
  toolState.set(toolName, true);
});

/**
 * GET /api/tools
 * Get all available tools with their enabled state
 */
router.get('/', (req: Request, res: Response) => {
  const toolsList = Object.entries(tools).map(([name, tool]) => ({
    name,
    description: tool.definition.description,
    enabled: toolState.get(name) ?? true,
  }));

  res.json({ tools: toolsList });
});

/**
 * PATCH /api/tools/:name
 * Enable or disable a specific tool
 */
router.patch('/:name', (req: Request, res: Response) => {
  const { name } = req.params;
  const { enabled } = req.body;

  if (!tools[name]) {
    return res.status(404).json({ error: `Tool '${name}' not found` });
  }

  if (typeof enabled !== 'boolean') {
    return res.status(400).json({ error: 'enabled must be a boolean' });
  }

  toolState.set(name, enabled);

  res.json({
    success: true,
    tool: name,
    enabled,
  });
});

/**
 * Get list of enabled tools
 */
export function getEnabledTools() {
  return Object.entries(tools)
    .filter(([name]) => toolState.get(name) !== false)
    .map(([, tool]) => tool.definition);
}

/**
 * Check if a tool is enabled
 */
export function isToolEnabled(toolName: string): boolean {
  return toolState.get(toolName) !== false;
}

export default router;
