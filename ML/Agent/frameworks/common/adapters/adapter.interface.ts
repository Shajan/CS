import type {
  AgentRequest,
  AgentResponse,
  ConversationHistory,
  AdapterCapabilities,
} from '../types/index.js';

/**
 * AgentAdapter Interface
 *
 * All framework adapters must implement this interface to work with
 * the unified API layer. This provides a framework-agnostic abstraction
 * for chat, history management, and metadata.
 *
 * @example
 * ```typescript
 * class MyAdapter implements AgentAdapter {
 *   readonly name = 'my-framework';
 *   readonly displayName = 'My Framework';
 *   readonly version = '1.0.0';
 *
 *   async chat(request: AgentRequest): Promise<AgentResponse> {
 *     // Implementation
 *   }
 *
 *   async clearHistory(sessionId: string): Promise<void> {
 *     // Implementation
 *   }
 *
 *   async getHistory(sessionId: string): Promise<ConversationHistory> {
 *     // Implementation
 *   }
 * }
 * ```
 */
export interface AgentAdapter {
  // ============ Metadata (REQUIRED) ============

  /** Machine-readable name (e.g., "claude-agent", "langchain") */
  readonly name: string;

  /** Human-readable display name (e.g., "Claude Agent SDK", "LangChain") */
  readonly displayName: string;

  /** Adapter version */
  readonly version: string;

  /** Brief description of the framework */
  readonly description?: string;

  // ============ Core Methods (REQUIRED) ============

  /**
   * Send a message to the agent and get a response
   *
   * @param request - Agent request containing message and session info
   * @returns Agent response with the assistant's message
   */
  chat(request: AgentRequest): Promise<AgentResponse>;

  /**
   * Clear conversation history for a session
   *
   * @param sessionId - Session identifier
   */
  clearHistory(sessionId: string): Promise<void>;

  /**
   * Get conversation history for a session
   *
   * @param sessionId - Session identifier
   * @returns Conversation history with all messages
   */
  getHistory(sessionId: string): Promise<ConversationHistory>;

  // ============ Lifecycle Methods (OPTIONAL) ============

  /**
   * Initialize the adapter with configuration
   *
   * Called once when the adapter is registered. Use this to:
   * - Validate API keys
   * - Set up clients
   * - Initialize resources
   *
   * @param config - Configuration object
   */
  initialize?(config: Record<string, any>): Promise<void>;

  /**
   * Shutdown the adapter and clean up resources
   *
   * Called when the server is shutting down. Use this to:
   * - Close connections
   * - Clean up resources
   * - Save state if needed
   */
  shutdown?(): Promise<void>;

  /**
   * Check if the adapter is healthy and ready
   *
   * @returns true if the adapter is operational, false otherwise
   */
  healthCheck?(): Promise<boolean>;

  // ============ Metadata Methods (OPTIONAL) ============

  /**
   * Get framework capabilities
   *
   * @returns Capabilities object describing what the framework supports
   */
  getCapabilities?(): AdapterCapabilities;

  /**
   * Get current configuration
   *
   * @returns Configuration object (without sensitive data like API keys)
   */
  getConfiguration?(): Record<string, any>;
}
