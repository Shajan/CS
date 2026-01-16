import type { ConversationHistory, Message } from '../types/index.js';
import type { AgentAdapter } from './adapter.interface.js';

/**
 * BaseAdapter
 *
 * Optional base class that provides common functionality for adapters.
 * Extends this to simplify adapter implementation.
 *
 * Features:
 * - Session management with Map-based storage
 * - History management
 * - Logging utilities
 */
export abstract class BaseAdapter implements Partial<AgentAdapter> {
  protected sessions: Map<string, ConversationHistory> = new Map();

  /**
   * Get or create a session
   */
  protected getOrCreateSession(sessionId: string): ConversationHistory {
    if (!this.sessions.has(sessionId)) {
      this.sessions.set(sessionId, {
        sessionId,
        messages: [],
        framework: this.name,
        startedAt: new Date(),
        lastActiveAt: new Date(),
      });
    }
    return this.sessions.get(sessionId)!;
  }

  /**
   * Add a message to session history
   */
  protected addMessage(sessionId: string, message: Message): void {
    const session = this.getOrCreateSession(sessionId);
    session.messages.push(message);
    session.lastActiveAt = new Date();
  }

  /**
   * Get conversation history for a session
   */
  async getHistory(sessionId: string): Promise<ConversationHistory> {
    return this.getOrCreateSession(sessionId);
  }

  /**
   * Clear conversation history for a session
   */
  async clearHistory(sessionId: string): Promise<void> {
    this.sessions.delete(sessionId);
  }

  /**
   * Log a message with adapter name prefix
   */
  protected log(message: string, ...args: any[]): void {
    console.log(`[${this.name}] ${message}`, ...args);
  }

  /**
   * Log an error with adapter name prefix
   */
  protected logError(message: string, error?: Error): void {
    console.error(`[${this.name}] ERROR: ${message}`, error);
  }

  // Abstract properties that must be implemented
  abstract readonly name: string;
  abstract readonly displayName: string;
  abstract readonly version: string;
}
