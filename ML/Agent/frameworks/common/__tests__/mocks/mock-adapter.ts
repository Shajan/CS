import type {
  AgentRequest,
  AgentResponse,
  ConversationHistory,
  AdapterCapabilities,
} from '../../types/index.js';
import type { AgentAdapter } from '../../adapters/index.js';
import { BaseAdapter } from '../../adapters/base-adapter.js';

/**
 * MockAdapter
 *
 * A simple mock adapter for testing purposes.
 * Returns predefined responses and tracks method calls.
 */
export class MockAdapter extends BaseAdapter implements AgentAdapter {
  readonly name = 'mock';
  readonly displayName = 'Mock Adapter';
  readonly version = '1.0.0';
  readonly description = 'A mock adapter for testing';

  // Track method calls for assertions
  public chatCalls: AgentRequest[] = [];
  public clearHistoryCalls: string[] = [];
  public getHistoryCalls: string[] = [];
  public initializeCalls: any[] = [];
  public shutdownCalls: number = 0;
  public healthCheckCalls: number = 0;

  // Configure mock responses
  public mockResponse: string = 'Mock response';
  public mockHealthy: boolean = true;
  public shouldThrowError: boolean = false;

  async chat(request: AgentRequest): Promise<AgentResponse> {
    this.chatCalls.push(request);

    if (this.shouldThrowError) {
      throw new Error('Mock adapter error');
    }

    // Add message to history
    this.addMessage(request.sessionId, {
      role: 'user',
      content: request.message,
      timestamp: new Date(),
    });

    this.addMessage(request.sessionId, {
      role: 'assistant',
      content: this.mockResponse,
      timestamp: new Date(),
    });

    return {
      response: this.mockResponse,
      sessionId: request.sessionId,
      framework: this.name,
      metadata: {
        model: 'mock-model-1.0',
        tokensUsed: 42,
        duration: 100,
      },
    };
  }

  async clearHistory(sessionId: string): Promise<void> {
    this.clearHistoryCalls.push(sessionId);
    await super.clearHistory(sessionId);
  }

  async getHistory(sessionId: string): Promise<ConversationHistory> {
    this.getHistoryCalls.push(sessionId);
    return super.getHistory(sessionId);
  }

  async initialize(config: Record<string, any>): Promise<void> {
    this.initializeCalls.push(config);
    this.log('Initialized with config:', config);
  }

  async shutdown(): Promise<void> {
    this.shutdownCalls++;
    this.log('Shutting down');
  }

  async healthCheck(): Promise<boolean> {
    this.healthCheckCalls++;
    return this.mockHealthy;
  }

  getCapabilities(): AdapterCapabilities {
    return {
      supportsStreaming: false,
      supportsTools: false,
      supportsMultiModal: false,
      supportsMultiAgent: false,
      supportsMemory: true,
      maxContextLength: 1000,
      supportedModels: ['mock-model-1.0'],
    };
  }

  getConfiguration(): Record<string, any> {
    return {
      model: 'mock-model-1.0',
      temperature: 0.7,
    };
  }

  // Helper method to reset call tracking
  resetCalls(): void {
    this.chatCalls = [];
    this.clearHistoryCalls = [];
    this.getHistoryCalls = [];
    this.initializeCalls = [];
    this.shutdownCalls = 0;
    this.healthCheckCalls = 0;
  }
}
