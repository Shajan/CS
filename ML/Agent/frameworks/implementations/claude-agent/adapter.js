import { ToolUsingAgent } from './agents/tool-using-agent.js';
import config from './config.json' with { type: 'json' };
/**
 * ClaudeAgentAdapter
 *
 * Adapter for Anthropic's Claude Agent SDK. Wraps the existing
 * ToolUsingAgent implementation to conform to the AgentAdapter interface.
 */
export class ClaudeAgentAdapter {
    name = config.name;
    displayName = config.displayName;
    version = config.version;
    description = config.description;
    agents = new Map();
    apiKey;
    constructor(apiKey) {
        this.apiKey = apiKey || process.env.ANTHROPIC_API_KEY || '';
        if (!this.apiKey) {
            throw new Error('ANTHROPIC_API_KEY is required for Claude Agent adapter');
        }
    }
    /**
     * Get or create an agent for a session
     */
    getAgent(sessionId, requestConfig) {
        if (!this.agents.has(sessionId)) {
            const agent = new ToolUsingAgent({
                apiKey: this.apiKey,
                model: requestConfig?.model || config.defaultConfig.model,
                maxTokens: requestConfig?.maxTokens || config.defaultConfig.maxTokens,
                sessionId,
            });
            this.agents.set(sessionId, agent);
        }
        return this.agents.get(sessionId);
    }
    /**
     * Send a message to Claude and get a response
     */
    async chat(request) {
        const startTime = Date.now();
        const agent = this.getAgent(request.sessionId, request.config);
        try {
            const response = await agent.chat(request.message);
            const duration = Date.now() - startTime;
            return {
                response,
                sessionId: request.sessionId,
                framework: this.name,
                metadata: {
                    model: request.config?.model || config.defaultConfig.model,
                    duration,
                },
            };
        }
        catch (error) {
            console.error(`[${this.name}] Error during chat:`, error);
            throw error;
        }
    }
    /**
     * Clear conversation history for a session
     */
    async clearHistory(sessionId) {
        const agent = this.agents.get(sessionId);
        if (agent) {
            agent.clearHistory();
        }
        // Also remove the agent instance to free memory
        this.agents.delete(sessionId);
    }
    /**
     * Get conversation history for a session
     */
    async getHistory(sessionId) {
        const agent = this.agents.get(sessionId);
        if (!agent) {
            return {
                sessionId,
                messages: [],
                framework: this.name,
            };
        }
        const messages = agent.getHistory().map((msg) => ({
            role: msg.role,
            content: msg.content,
        }));
        return {
            sessionId,
            messages,
            framework: this.name,
        };
    }
    /**
     * Initialize the adapter
     */
    async initialize(initConfig) {
        if (initConfig.apiKey) {
            this.apiKey = initConfig.apiKey;
        }
        console.log(`✓ Initialized ${this.displayName}`);
    }
    /**
     * Shutdown and cleanup
     */
    async shutdown() {
        this.agents.clear();
        console.log(`✓ Shut down ${this.displayName}`);
    }
    /**
     * Health check
     */
    async healthCheck() {
        return !!this.apiKey;
    }
    /**
     * Get framework capabilities
     */
    getCapabilities() {
        return config.capabilities;
    }
    /**
     * Get current configuration (without sensitive data)
     */
    getConfiguration() {
        return {
            model: config.defaultConfig.model,
            maxTokens: config.defaultConfig.maxTokens,
            hasApiKey: !!this.apiKey,
        };
    }
}
//# sourceMappingURL=adapter.js.map