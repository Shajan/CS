import { Router, type Request, type Response } from 'express';
import { settingsStorage, type MCPServer } from '../../storage/settings-storage.js';

const router = Router();

/**
 * GET /api/settings
 * Get all settings
 */
router.get('/', (req: Request, res: Response) => {
  try {
    const settings = settingsStorage.getSettings();
    res.json(settings);
  } catch (error) {
    console.error('Error in GET /api/settings:', error);
    res.status(500).json({
      error: 'Failed to get settings',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/settings/framework/:name
 * Get settings for a specific framework
 */
router.get('/framework/:name', (req: Request, res: Response) => {
  try {
    const { name } = req.params;
    const settings = settingsStorage.getFrameworkSettings(name);
    res.json({ framework: name, settings });
  } catch (error) {
    console.error(`Error in GET /api/settings/framework/${req.params.name}:`, error);
    res.status(500).json({
      error: 'Failed to get framework settings',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * PUT /api/settings/framework/:name
 * Update settings for a specific framework
 */
router.put('/framework/:name', (req: Request, res: Response) => {
  try {
    const { name } = req.params;
    const config = req.body;

    settingsStorage.setFrameworkSettings(name, config);
    res.json({ success: true, framework: name, settings: config });
  } catch (error) {
    console.error(`Error in PUT /api/settings/framework/${req.params.name}:`, error);
    res.status(500).json({
      error: 'Failed to update framework settings',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/settings/mcp
 * Get all MCP servers
 */
router.get('/mcp', (req: Request, res: Response) => {
  try {
    const servers = settingsStorage.getMCPServers();
    res.json({ servers });
  } catch (error) {
    console.error('Error in GET /api/settings/mcp:', error);
    res.status(500).json({
      error: 'Failed to get MCP servers',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * POST /api/settings/mcp
 * Add a new MCP server
 */
router.post('/mcp', (req: Request, res: Response) => {
  try {
    const server: MCPServer = req.body;

    if (!server.name || !server.command) {
      res.status(400).json({ error: 'Name and command are required' });
      return;
    }

    settingsStorage.addMCPServer(server);
    res.json({ success: true, server });
  } catch (error) {
    console.error('Error in POST /api/settings/mcp:', error);
    res.status(500).json({
      error: 'Failed to add MCP server',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * PUT /api/settings/mcp/:name
 * Update an existing MCP server
 */
router.put('/mcp/:name', (req: Request, res: Response) => {
  try {
    const { name } = req.params;
    const updates = req.body;

    settingsStorage.updateMCPServer(name, updates);
    res.json({ success: true, name, updates });
  } catch (error) {
    console.error(`Error in PUT /api/settings/mcp/${req.params.name}:`, error);
    res.status(500).json({
      error: 'Failed to update MCP server',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * DELETE /api/settings/mcp/:name
 * Remove an MCP server
 */
router.delete('/mcp/:name', (req: Request, res: Response) => {
  try {
    const { name } = req.params;

    settingsStorage.removeMCPServer(name);
    res.json({ success: true, name });
  } catch (error) {
    console.error(`Error in DELETE /api/settings/mcp/${req.params.name}:`, error);
    res.status(500).json({
      error: 'Failed to remove MCP server',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

export default router;
