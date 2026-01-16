import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { AgentAdapter } from '../../common/adapters/adapter.interface.js';
import { AgentRequest, AgentResponse, AdapterCapabilities } from '../../common/types/agent.types.js';
import { ConversationHistory } from '../../common/types/message.types.js';
import config from './config.json' with { type: 'json' };

/**
 * LangChain adapter implementation
 */
export class LangChainAdapter implements AgentAdapter {
  readonly name = 'langchain';
  readonly displayName = 'LangChain';
  readonly version = '1.0.0';
  readonly description = 'Popular Python/JS framework for building LLM applications with chains and agents';

  private models: Map<string, ChatOpenAI> = new Map();
  private histories: Map<string, BaseMessage[]> = new Map();
  private apiKey: string;

  constructor() {
    // Get API key from environment
    this.apiKey = process.env.OPENAI_API_KEY || '';
    if (!this.apiKey) {
      throw new Error('OPENAI_API_KEY is required for LangChain adapter');
    }
  }

  /**
   * Initialize the adapter
   */
  async initialize(initConfig: Record<string, any> = {}): Promise<void> {
    console.log('✓ Initialized LangChain');
    console.log(`✓ Registered adapter: ${this.displayName} (${this.name})`);
  }

  /**
   * Send a chat message
   */
  async chat(request: AgentRequest): Promise<AgentResponse> {
    const startTime = Date.now();

    // Get or create model for this session
    const model = this.getModel(request.sessionId, request.config);

    // Get conversation history
    const history = this.getHistoryMessages(request.sessionId);

    // Add user message to history
    const userMessage = new HumanMessage(request.message);
    history.push(userMessage);

    try {
      // Send messages to LangChain
      const response = await model.invoke(history);

      // Add assistant message to history
      history.push(response);

      const duration = Date.now() - startTime;

      return {
        response: response.content as string,
        sessionId: request.sessionId,
        framework: this.name,
        metadata: {
          model: config.defaultConfig.model,
          duration,
        },
      };
    } catch (error) {
      throw new Error(`LangChain chat failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Clear conversation history for a session
   */
  async clearHistory(sessionId: string): Promise<void> {
    this.histories.delete(sessionId);
    this.models.delete(sessionId);
  }

  /**
   * Get conversation history for a session
   */
  async getHistory(sessionId: string): Promise<ConversationHistory> {
    const messages = this.histories.get(sessionId) || [];

    return {
      sessionId,
      messages: messages.map((msg) => ({
        role: msg instanceof HumanMessage ? 'user' : msg instanceof AIMessage ? 'assistant' : 'system',
        content: msg.content as string,
      })),
      framework: this.name,
    };
  }

  /**
   * Get adapter capabilities
   */
  getCapabilities(): AdapterCapabilities {
    return config.capabilities;
  }

  /**
   * Get adapter configuration
   */
  getConfiguration(): Record<string, any> {
    return config.defaultConfig;
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      // Try to create a model to verify API key works
      const testModel = new ChatOpenAI({
        openAIApiKey: this.apiKey,
        modelName: config.defaultConfig.model,
      });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get or create ChatOpenAI model for a session
   */
  private getModel(sessionId: string, requestConfig?: Record<string, any>): ChatOpenAI {
    let model = this.models.get(sessionId);

    if (!model) {
      model = new ChatOpenAI({
        openAIApiKey: this.apiKey,
        modelName: requestConfig?.model || config.defaultConfig.model,
        temperature: requestConfig?.temperature ?? config.defaultConfig.temperature,
        maxTokens: requestConfig?.maxTokens ?? config.defaultConfig.maxTokens,
      });

      this.models.set(sessionId, model);
    }

    return model;
  }

  /**
   * Get conversation history messages for a session
   */
  private getHistoryMessages(sessionId: string): BaseMessage[] {
    let history = this.histories.get(sessionId);

    if (!history) {
      history = [];
      this.histories.set(sessionId, history);
    }

    return history;
  }
}
