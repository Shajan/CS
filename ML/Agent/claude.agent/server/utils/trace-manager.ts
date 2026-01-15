import { Response } from 'express';

export interface TraceEvent {
  timestamp: string;
  category: string;
  data: any;
  sessionId?: string;
}

/**
 * TraceManager - Collects and broadcasts agent traces to connected clients
 */
class TraceManager {
  private clients: Set<Response> = new Set();
  private traceHistory: TraceEvent[] = [];
  private maxHistory = 100;

  /**
   * Add a new SSE client
   */
  addClient(res: Response): void {
    this.clients.add(res);

    // Send existing trace history to new client
    this.traceHistory.forEach((trace) => {
      this.sendToClient(res, trace);
    });
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
  trace(category: string, data: any, sessionId?: string): void {
    const event: TraceEvent = {
      timestamp: new Date().toISOString(),
      category,
      data,
      sessionId,
    };

    // Add to history
    this.traceHistory.push(event);
    if (this.traceHistory.length > this.maxHistory) {
      this.traceHistory.shift();
    }

    // Broadcast to all connected clients
    this.broadcast(event);

    // Also log to console
    console.log('\n' + '='.repeat(80));
    console.log(`[TRACE - ${category}] ${event.timestamp}`);
    if (sessionId) console.log(`Session: ${sessionId}`);
    console.log('='.repeat(80));
    console.log(JSON.stringify(data, null, 2));
    console.log('='.repeat(80) + '\n');
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
   * Get trace history
   */
  getHistory(): TraceEvent[] {
    return [...this.traceHistory];
  }

  /**
   * Clear trace history
   */
  clearHistory(): void {
    this.traceHistory = [];
  }
}

// Export singleton instance
export const traceManager = new TraceManager();
