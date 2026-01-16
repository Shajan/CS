import { Router, type Request, type Response } from 'express';
import { traceStorage } from '../../storage/trace-storage.js';

const router = Router();

/**
 * GET /api/traces?sessionId=xxx
 * Get trace events for a session
 */
router.get('/', async (req: Request, res: Response) => {
  try {
    const { sessionId } = req.query;

    if (!sessionId || typeof sessionId !== 'string') {
      res.status(400).json({ error: 'SessionId query parameter is required' });
      return;
    }

    const traces = traceStorage.getTraces(sessionId);

    res.json({
      sessionId,
      traces,
      count: traces.length,
    });
  } catch (error) {
    console.error('Error in GET /api/traces:', error);
    res.status(500).json({
      error: 'Failed to get traces',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * DELETE /api/traces?sessionId=xxx
 * Clear trace events for a session
 */
router.delete('/', async (req: Request, res: Response) => {
  try {
    const { sessionId } = req.query;

    if (!sessionId || typeof sessionId !== 'string') {
      res.status(400).json({ error: 'SessionId query parameter is required' });
      return;
    }

    traceStorage.clearTraces(sessionId);

    res.json({
      success: true,
      sessionId,
    });
  } catch (error) {
    console.error('Error in DELETE /api/traces:', error);
    res.status(500).json({
      error: 'Failed to clear traces',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

/**
 * GET /api/traces/sessions
 * Get all session IDs with traces
 */
router.get('/sessions', async (req: Request, res: Response) => {
  try {
    const sessions = traceStorage.getAllSessions();
    res.json({ sessions, count: sessions.length });
  } catch (error) {
    console.error('Error in GET /api/traces/sessions:', error);
    res.status(500).json({
      error: 'Failed to get sessions',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

export default router;
