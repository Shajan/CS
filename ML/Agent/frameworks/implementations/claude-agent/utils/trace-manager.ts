import { Response } from 'express';
import { traceStorage } from '../../../common/storage/trace-storage.js';
import type { TraceEvent, TraceCategory } from '../../../common/types/trace.types.js';

/**
 * TraceManager - Collects and broadcasts agent traces to connected clients
 * Now uses the common trace storage system
 */
class TraceManager {
  private clients: Set<Response> = new Set();
  private framework = 'claude-agent';

  /**
   * Add a new SSE client
   * Note: History is not sent here - clients should fetch it via /history endpoint
   */
  addClient(res: Response): void {
    this.clients.add(res);
  }

  /**
   * Remove an SSE client
   */
  removeClient(res: Response): void {
    this.clients.delete(res);
  }

  /**
   * Log a trace event and broadcast to all clients
   */
  trace(category: string, data: any, sessionId: string = 'default'): void {
    const event: TraceEvent = {
      timestamp: new Date().toISOString(),
      sessionId,
      framework: this.framework,
      category: category as TraceCategory,
      data,
    };

    // Add to common trace storage
    traceStorage.addTrace(event);

    // Broadcast to all connected clients
    this.broadcast(event);

    // Also log to console in development
    if (process.env.NODE_ENV !== 'production') {
      console.log('\n' + '='.repeat(80));
      console.log(`[TRACE - ${category}] ${event.timestamp}`);
      console.log(`Session: ${sessionId} | Framework: ${this.framework}`);
      console.log('='.repeat(80));
      console.log(JSON.stringify(data, null, 2));
      console.log('='.repeat(80) + '\n');
    }
  }

  /**
   * Broadcast a trace event to all connected clients
   */
  private broadcast(event: TraceEvent): void {
    this.clients.forEach((client) => {
      this.sendToClient(client, event);
    });
  }

  /**
   * Send a trace event to a specific client
   */
  private sendToClient(client: Response, event: TraceEvent): void {
    try {
      client.write(`data: ${JSON.stringify(event)}\n\n`);
    } catch (error) {
      // Client disconnected, remove it
      this.clients.delete(client);
    }
  }

  /**
   * Get trace history for a session
   */
  getHistory(sessionId: string): TraceEvent[] {
    return traceStorage.getTraces(sessionId, this.framework);
  }

  /**
   * Clear trace history for a session
   */
  clearHistory(sessionId: string): void {
    traceStorage.clearTraces(sessionId, this.framework);
  }
}

// Export singleton instance
export const traceManager = new TraceManager();
