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
declare class TraceManager {
    private clients;
    private traceHistory;
    private maxHistory;
    /**
     * Add a new SSE client
     * Note: History is not sent here - clients should fetch it via /history endpoint
     */
    addClient(res: Response): void;
    /**
     * Remove an SSE client
     */
    removeClient(res: Response): void;
    /**
     * Log a trace event and broadcast to all clients
     */
    trace(category: string, data: any, sessionId?: string): void;
    /**
     * Broadcast a trace event to all connected clients
     */
    private broadcast;
    /**
     * Send a trace event to a specific client
     */
    private sendToClient;
    /**
     * Get trace history
     */
    getHistory(): TraceEvent[];
    /**
     * Clear trace history
     */
    clearHistory(): void;
}
export declare const traceManager: TraceManager;
export {};
//# sourceMappingURL=trace-manager.d.ts.map