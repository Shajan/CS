/**
 * Message role in a conversation
 */
export type MessageRole = 'user' | 'assistant' | 'system';

/**
 * Message structure for agent conversations
 */
export interface Message {
  role: MessageRole;
  content: string;
  timestamp?: Date;
  metadata?: Record<string, any>;
}

/**
 * Conversation history for a session
 */
export interface ConversationHistory {
  sessionId: string;
  messages: Message[];
  framework?: string;
  startedAt?: Date;
  lastActiveAt?: Date;
}
