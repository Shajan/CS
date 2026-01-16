/**
 * In-memory storage for trace events
 */

import type { TraceEvent, TraceStorage } from '../types/trace.types.js';

export class InMemoryTraceStorage implements TraceStorage {
  private traces: Map<string, TraceEvent[]> = new Map();

  addTrace(trace: TraceEvent): void {
    const key = this.getKey(trace.sessionId, trace.framework);
    const existingTraces = this.traces.get(key) || [];
    existingTraces.push(trace);
    this.traces.set(key, existingTraces);
  }

  getTraces(sessionId: string, framework?: string): TraceEvent[] {
    if (framework) {
      const key = this.getKey(sessionId, framework);
      return this.traces.get(key) || [];
    }

    // If no framework specified, return all traces for this session
    const allTraces: TraceEvent[] = [];
    for (const [key, traces] of this.traces.entries()) {
      if (key.startsWith(`${sessionId}:`)) {
        allTraces.push(...traces);
      }
    }
    return allTraces.sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }

  clearTraces(sessionId: string, framework?: string): void {
    if (framework) {
      const key = this.getKey(sessionId, framework);
      this.traces.delete(key);
    } else {
      // Clear all traces for this session
      const keysToDelete: string[] = [];
      for (const key of this.traces.keys()) {
        if (key.startsWith(`${sessionId}:`)) {
          keysToDelete.push(key);
        }
      }
      keysToDelete.forEach(key => this.traces.delete(key));
    }
  }

  getAllSessions(): string[] {
    const sessions = new Set<string>();
    for (const key of this.traces.keys()) {
      const sessionId = key.split(':')[0];
      sessions.add(sessionId);
    }
    return Array.from(sessions);
  }

  private getKey(sessionId: string, framework: string): string {
    return `${sessionId}:${framework}`;
  }
}

// Export singleton instance
export const traceStorage = new InMemoryTraceStorage();
