import Anthropic from '@anthropic-ai/sdk';
import { traceManager } from '../utils/trace-manager.js';
/**
 * Base Agent class - provides a simple abstraction for interacting with Claude
 *
 * This class handles the core interaction with the Anthropic API,
 * making it easy to build custom agents by extending this class.
 */
export class BaseAgent {
    client;
    model;
    maxTokens;
    conversationHistory = [];
    sessionId;
    constructor(config) {
        this.client = new Anthropic({
            apiKey: config.apiKey,
        });
        this.model = config.model || 'claude-sonnet-4-5-20250929';
        this.maxTokens = config.maxTokens || 1024;
        this.sessionId = config.sessionId || 'default';
    }
    /**
     * Log trace information to the trace manager
     */
    trace(category, data) {
        traceManager.trace(category, data, this.sessionId);
    }
    /**
     * Send a message to Claude and get a response
     */
    async chat(userMessage) {
        // Add user message to history
        this.conversationHistory.push({
            role: 'user',
            content: userMessage,
        });
        this.trace('USER_MESSAGE', {
            message: userMessage,
            conversationLength: this.conversationHistory.length,
        });
        // Prepare API request
        const apiRequest = {
            model: this.model,
            max_tokens: this.maxTokens,
            messages: this.conversationHistory,
        };
        this.trace('API_REQUEST', apiRequest);
        // Call Claude API
        const startTime = Date.now();
        const response = await this.client.messages.create(apiRequest);
        const duration = Date.now() - startTime;
        this.trace('API_RESPONSE', {
            id: response.id,
            model: response.model,
            role: response.role,
            stop_reason: response.stop_reason,
            usage: response.usage,
            duration_ms: duration,
            content: response.content,
        });
        // Extract assistant's response
        const assistantMessage = response.content[0].type === 'text'
            ? response.content[0].text
            : '';
        // Add assistant's response to history
        this.conversationHistory.push({
            role: 'assistant',
            content: assistantMessage,
        });
        this.trace('ASSISTANT_RESPONSE', {
            message: assistantMessage,
            messageLength: assistantMessage.length,
            tokensUsed: response.usage,
        });
        return assistantMessage;
    }
    /**
     * Clear the conversation history
     */
    clearHistory() {
        this.conversationHistory = [];
    }
    /**
     * Get the current conversation history
     */
    getHistory() {
        return [...this.conversationHistory];
    }
}
//# sourceMappingURL=base-agent.js.map