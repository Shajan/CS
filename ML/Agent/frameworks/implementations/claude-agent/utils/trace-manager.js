/**
 * TraceManager - Collects and broadcasts agent traces to connected clients
 */
class TraceManager {
    clients = new Set();
    traceHistory = [];
    maxHistory = 100;
    /**
     * Add a new SSE client
     * Note: History is not sent here - clients should fetch it via /history endpoint
     */
    addClient(res) {
        this.clients.add(res);
    }
    /**
     * Remove an SSE client
     */
    removeClient(res) {
        this.clients.delete(res);
    }
    /**
     * Log a trace event and broadcast to all clients
     */
    trace(category, data, sessionId) {
        const event = {
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
        if (sessionId)
            console.log(`Session: ${sessionId}`);
        console.log('='.repeat(80));
        console.log(JSON.stringify(data, null, 2));
        console.log('='.repeat(80) + '\n');
    }
    /**
     * Broadcast a trace event to all connected clients
     */
    broadcast(event) {
        this.clients.forEach((client) => {
            this.sendToClient(client, event);
        });
    }
    /**
     * Send a trace event to a specific client
     */
    sendToClient(client, event) {
        try {
            client.write(`data: ${JSON.stringify(event)}\n\n`);
        }
        catch (error) {
            // Client disconnected, remove it
            this.clients.delete(client);
        }
    }
    /**
     * Get trace history
     */
    getHistory() {
        return [...this.traceHistory];
    }
    /**
     * Clear trace history
     */
    clearHistory() {
        this.traceHistory = [];
    }
}
// Export singleton instance
export const traceManager = new TraceManager();
//# sourceMappingURL=trace-manager.js.map