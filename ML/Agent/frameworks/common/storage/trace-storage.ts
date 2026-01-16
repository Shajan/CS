/**
 * In-memory storage for trace events
 */

import type { TraceEvent, TraceStorage } from '../types/trace.types.js';

export class InMemoryTraceStorage implements TraceStorage {
  private traces: Map<string, TraceEvent[]> = new Map();

  addTrace(trace: TraceEvent): void {
    const existingTraces = this.traces.get(trace.sessionId) || [];
    existingTraces.push(trace);
    this.traces.set(trace.sessionId, existingTraces);
  }

  getTraces(sessionId: string, framework?: string): TraceEvent[] {
    const allTraces = this.traces.get(sessionId) || [];
    return allTraces.sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }

  clearTraces(sessionId: string, framework?: string): void {
    this.traces.delete(sessionId);
  }

  getAllSessions(): string[] {
    return Array.from(this.traces.keys());
  }
}

// Export singleton instance
export const traceStorage = new InMemoryTraceStorage();
