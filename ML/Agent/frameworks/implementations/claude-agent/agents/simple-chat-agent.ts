import { BaseAgent, AgentConfig } from './base-agent.js';

/**
 * SimpleChatAgent - A straightforward conversational agent
 *
 * This agent extends BaseAgent and adds a simple system prompt
 * to demonstrate how to create custom agents with specific behaviors.
 */
export class SimpleChatAgent extends BaseAgent {
  private systemPrompt: string;

  constructor(config: AgentConfig, systemPrompt?: string) {
    super(config);
    this.systemPrompt = systemPrompt || 'You are a helpful AI assistant.';
  }

  /**
   * Chat with the agent using a system prompt
   */
  async chat(userMessage: string): Promise<string> {
    // For the first message, prepend the system prompt
    if (this.conversationHistory.length === 0) {
      const messageWithSystem = `${this.systemPrompt}\n\nUser: ${userMessage}`;
      return super.chat(messageWithSystem);
    }

    return super.chat(userMessage);
  }
}
