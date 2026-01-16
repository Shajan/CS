import type {
  AgentRequest,
  AgentResponse,
  ConversationHistory,
  AdapterCapabilities,
} from '../../common/types/index.js';
import type { AgentAdapter } from '../../common/adapters/index.js';
import { ToolUsingAgent } from './agents/tool-using-agent.js';
import config from './config.json' with { type: 'json' };

/**
 * ClaudeAgentAdapter
 *
 * Adapter for Anthropic's Claude Agent SDK. Wraps the existing
 * ToolUsingAgent implementation to conform to the AgentAdapter interface.
 */
export class ClaudeAgentAdapter implements AgentAdapter {
  readonly name = config.name;
  readonly displayName = config.displayName;
  readonly version = config.version;
  readonly description = config.description;

  private agents: Map<string, ToolUsingAgent> = new Map();
  private apiKey: string;

  constructor(apiKey?: string) {
    this.apiKey = apiKey || process.env.ANTHROPIC_API_KEY || '';
    if (!this.apiKey) {
      throw new Error('ANTHROPIC_API_KEY is required for Claude Agent adapter');
    }
  }

  /**
   * Get or create an agent for a session
   */
  private getAgent(sessionId: string, requestConfig?: any): ToolUsingAgent {
    if (!this.agents.has(sessionId)) {
      const agent = new ToolUsingAgent({
        apiKey: this.apiKey,
        model: requestConfig?.model || config.defaultConfig.model,
        maxTokens: requestConfig?.maxTokens || config.defaultConfig.maxTokens,
        sessionId,
      });
      this.agents.set(sessionId, agent);
    }
    return this.agents.get(sessionId)!;
  }

  /**
   * Send a message to Claude and get a response
   */
  async chat(request: AgentRequest): Promise<AgentResponse> {
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
    } catch (error) {
      console.error(`[${this.name}] Error during chat:`, error);
      throw error;
    }
  }

  /**
   * Clear conversation history for a session
   */
  async clearHistory(sessionId: string): Promise<void> {
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
  async getHistory(sessionId: string): Promise<ConversationHistory> {
    const agent = this.agents.get(sessionId);
    if (!agent) {
      return {
        sessionId,
        messages: [],
        framework: this.name,
      };
    }

    const messages = agent.getHistory().map((msg) => ({
      role: msg.role as 'user' | 'assistant',
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
  async initialize(initConfig: Record<string, any>): Promise<void> {
    if (initConfig.apiKey) {
      this.apiKey = initConfig.apiKey;
    }
    console.log(`✓ Initialized ${this.displayName}`);
  }

  /**
   * Shutdown and cleanup
   */
  async shutdown(): Promise<void> {
    this.agents.clear();
    console.log(`✓ Shut down ${this.displayName}`);
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    return !!this.apiKey;
  }

  /**
   * Get framework capabilities
   */
  getCapabilities(): AdapterCapabilities {
    return config.capabilities;
  }

  /**
   * Get current configuration (without sensitive data)
   */
  getConfiguration(): Record<string, any> {
    return {
      model: config.defaultConfig.model,
      maxTokens: config.defaultConfig.maxTokens,
      hasApiKey: !!this.apiKey,
    };
  }
}
