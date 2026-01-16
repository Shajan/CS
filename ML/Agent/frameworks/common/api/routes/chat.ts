import { Router, type Request, type Response } from 'express';
import type { AgentAdapter } from '../../adapters/adapter.interface.js';
import type { AgentRequest } from '../../types/index.js';
import { frameworkSelector } from '../middleware/framework-selector.js';

const router = Router();

/**
 * POST /api/chat
 * Send a message to an agent framework
 */
router.post('/', frameworkSelector, async (req: Request, res: Response) => {
  try {
    const adapter = (req as any).adapter as AgentAdapter;
    const { message, sessionId, config, metadata } = req.body;

    // Validate request
    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Message is required and must be a string' });
      return;
    }

    if (!sessionId || typeof sessionId !== 'string') {
      res.status(400).json({ error: 'SessionId is required and must be a string' });
      return;
    }

    // Build request
    const agentRequest: AgentRequest = {
      message,
      sessionId,
      framework: adapter.name,
      config,
      metadata,
    };

    // Call adapter
    const response = await adapter.chat(agentRequest);

    res.json(response);
  } catch (error) {
    console.error('Error in /api/chat:', error);
    res.status(500).json({
      error: 'Failed to process chat request',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

export default router;
