import type { AgentRequest, AgentResponse, ConversationHistory, AdapterCapabilities } from '../../common/types/index.js';
import type { AgentAdapter } from '../../common/adapters/index.js';
/**
 * ClaudeAgentAdapter
 *
 * Adapter for Anthropic's Claude Agent SDK. Wraps the existing
 * ToolUsingAgent implementation to conform to the AgentAdapter interface.
 */
export declare class ClaudeAgentAdapter implements AgentAdapter {
    readonly name: string;
    readonly displayName: string;
    readonly version: string;
    readonly description: string;
    private agents;
    private apiKey;
    constructor(apiKey?: string);
    /**
     * Get or create an agent for a session
     */
    private getAgent;
    /**
     * Send a message to Claude and get a response
     */
    chat(request: AgentRequest): Promise<AgentResponse>;
    /**
     * Clear conversation history for a session
     */
    clearHistory(sessionId: string): Promise<void>;
    /**
     * Get conversation history for a session
     */
    getHistory(sessionId: string): Promise<ConversationHistory>;
    /**
     * Initialize the adapter
     */
    initialize(initConfig: Record<string, any>): Promise<void>;
    /**
     * Shutdown and cleanup
     */
    shutdown(): Promise<void>;
    /**
     * Health check
     */
    healthCheck(): Promise<boolean>;
    /**
     * Get framework capabilities
     */
    getCapabilities(): AdapterCapabilities;
    /**
     * Get current configuration (without sensitive data)
     */
    getConfiguration(): Record<string, any>;
}
//# sourceMappingURL=adapter.d.ts.map