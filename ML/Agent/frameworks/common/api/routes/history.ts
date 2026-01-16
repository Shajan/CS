import { Router, type Request, type Response } from 'express';
import type { AgentAdapter } from '../../adapters/adapter.interface.js';
import { frameworkSelector } from '../middleware/framework-selector.js';

const router = Router();

/**
 * GET /api/history?sessionId=xxx&framework=xxx
 * Get conversation history for a session
 */
router.get('/', async (req: Request, res: Response) => {
  try {
    const { sessionId, framework = 'claude-agent' } = req.query;

    if (!sessionId || typeof sessionId !== 'string') {
      res.status(400).json({ error: 'SessionId query parameter is required' });
      return;
    }

    // Attach framework to body for middleware
    req.body.framework = framework;

    // Get adapter
    const { adapterRegistry } = await import('../../adapters/adapter-registry.js');
    const adapter = adapterRegistry.getAdapter(framework as string);

    const history = await adapter.getHistory(sessionId);
    res.json(history);
  } catch (error) {
    console.error('Error in GET /api/history:', error);
    res.status(500).json({
      error: 'Failed to get history',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * DELETE /api/history?sessionId=xxx&framework=xxx
 * Clear conversation history for a session
 */
router.delete('/', async (req: Request, res: Response) => {
  try {
    const { sessionId, framework = 'claude-agent' } = req.query;

    if (!sessionId || typeof sessionId !== 'string') {
      res.status(400).json({ error: 'SessionId query parameter is required' });
      return;
    }

    // Get adapter
    const { adapterRegistry } = await import('../../adapters/adapter-registry.js');
    const adapter = adapterRegistry.getAdapter(framework as string);

    await adapter.clearHistory(sessionId);
    res.json({ success: true, sessionId, framework });
  } catch (error) {
    console.error('Error in DELETE /api/history:', error);
    res.status(500).json({
      error: 'Failed to clear history',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

export default router;
