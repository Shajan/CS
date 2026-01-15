import { Router, Request, Response } from 'express';
import { ToolUsingAgent } from '../agents/index.js';

const router = Router();

// Store agent instances per session (in production, use proper session management)
const agents = new Map<string, ToolUsingAgent>();

/**
 * POST /api/agent/chat
 * Send a message to the agent and get a response
 */
router.post('/chat', async (req: Request, res: Response) => {
  try {
    const { message, sessionId = 'default' } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Get or create agent for this session
    let agent = agents.get(sessionId);
    if (!agent) {
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        return res.status(500).json({ error: 'API key not configured' });
      }

      agent = new ToolUsingAgent(
        { apiKey, sessionId },
        'You are a helpful AI assistant with access to tools. Use the tools available to help answer questions accurately.'
      );
      agents.set(sessionId, agent);
    }

    // Get response from agent
    const response = await agent.chat(message);

    res.json({
      response,
      sessionId,
    });
  } catch (error) {
    console.error('Agent error:', error);
    res.status(500).json({
      error: 'Failed to process message',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * POST /api/agent/clear
 * Clear the conversation history for a session
 */
router.post('/clear', (req: Request, res: Response) => {
  const { sessionId = 'default' } = req.body;

  const agent = agents.get(sessionId);
  if (agent) {
    agent.clearHistory();
  }

  res.json({ success: true, message: 'History cleared' });
});

/**
 * GET /api/agent/history
 * Get the conversation history for a session
 */
router.get('/history', (req: Request, res: Response) => {
  const sessionId = (req.query.sessionId as string) || 'default';

  const agent = agents.get(sessionId);
  const history = agent ? agent.getHistory() : [];

  res.json({ history });
});

export default router;
