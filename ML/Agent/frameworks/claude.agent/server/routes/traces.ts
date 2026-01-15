import { Router, Request, Response } from 'express';
import { traceManager } from '../utils/trace-manager.js';

const router = Router();

/**
 * GET /api/traces/stream
 * Server-Sent Events endpoint for streaming traces in real-time
 */
router.get('/stream', (req: Request, res: Response) => {
  // Set headers for SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');

  // Add this client to the trace manager
  traceManager.addClient(res);

  // Handle client disconnect
  req.on('close', () => {
    traceManager.removeClient(res);
  });
});

/**
 * GET /api/traces/history
 * Get the trace history
 */
router.get('/history', (req: Request, res: Response) => {
  const history = traceManager.getHistory();
  res.json({ history });
});

/**
 * POST /api/traces/clear
 * Clear the trace history
 */
router.post('/clear', (req: Request, res: Response) => {
  traceManager.clearHistory();
  res.json({ success: true, message: 'Trace history cleared' });
});

export default router;
