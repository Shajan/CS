import Anthropic from '@anthropic-ai/sdk';
/**
 * Message structure for agent conversations
 */
export interface Message {
    role: 'user' | 'assistant';
    content: string;
}
/**
 * Configuration for creating an agent
 */
export interface AgentConfig {
    apiKey: string;
    model?: string;
    maxTokens?: number;
    sessionId?: string;
}
/**
 * Base Agent class - provides a simple abstraction for interacting with Claude
 *
 * This class handles the core interaction with the Anthropic API,
 * making it easy to build custom agents by extending this class.
 */
export declare class BaseAgent {
    protected client: Anthropic;
    protected model: string;
    protected maxTokens: number;
    protected conversationHistory: Message[];
    protected sessionId: string;
    constructor(config: AgentConfig);
    /**
     * Log trace information to the trace manager
     */
    protected trace(category: string, data: any): void;
    /**
     * Send a message to Claude and get a response
     */
    chat(userMessage: string): Promise<string>;
    /**
     * Clear the conversation history
     */
    clearHistory(): void;
    /**
     * Get the current conversation history
     */
    getHistory(): Message[];
}
//# sourceMappingURL=base-agent.d.ts.map