import Anthropic from '@anthropic-ai/sdk';
import { BaseAgent, AgentConfig } from './base-agent.js';
import { executeTool } from '../tools/index.js';
import { getEnabledTools, isToolEnabled } from '../routes/tools.js';

/**
 * ToolUsingAgent - An agent that can use tools to perform actions
 *
 * This agent extends BaseAgent and adds the ability to:
 * - Call tools based on user requests
 * - Execute tools and return results to Claude
 * - Handle multi-turn tool interactions
 */
export class ToolUsingAgent extends BaseAgent {
  private systemPrompt: string;
  private maxToolRounds: number;

  constructor(config: AgentConfig, systemPrompt?: string, maxToolRounds: number = 5) {
    super(config);
    this.systemPrompt = systemPrompt || 'You are a helpful AI assistant with access to tools.';
    this.maxToolRounds = maxToolRounds;
  }

  /**
   * Chat with the agent using tools when needed
   */
  async chat(userMessage: string): Promise<string> {
    // Add user message to history
    this.conversationHistory.push({
      role: 'user',
      content: userMessage,
    });

    this.trace('USER_MESSAGE', {
      message: userMessage,
      conversationLength: this.conversationHistory.length,
    });

    let toolRound = 0;
    let finalResponse = '';

    // Tool calling loop
    while (toolRound < this.maxToolRounds) {
      // Get currently enabled tools
      const enabledTools = getEnabledTools();

      // Prepare API request with tools
      const apiRequest: any = {
        model: this.model,
        max_tokens: this.maxTokens,
        system: toolRound === 0 ? this.systemPrompt : undefined,
        messages: this.conversationHistory,
        tools: enabledTools,
      };

      this.trace('API_REQUEST', {
        ...apiRequest,
        toolRound,
      });

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
        toolRound,
      });

      // Check if Claude wants to use tools
      const toolUseBlocks = response.content.filter((block) => block.type === 'tool_use');

      if (toolUseBlocks.length === 0) {
        // No tools used, extract final text response
        const textBlock = response.content.find((block) => block.type === 'text');
        finalResponse = textBlock && textBlock.type === 'text' ? textBlock.text : '';

        // Add assistant's response to history
        this.conversationHistory.push({
          role: 'assistant',
          content: finalResponse,
        });

        this.trace('ASSISTANT_RESPONSE', {
          message: finalResponse,
          messageLength: finalResponse.length,
          tokensUsed: response.usage,
          toolRound,
        });

        break;
      }

      // Claude wants to use tools - add assistant message to history
      // We need to store the full content including tool_use blocks
      this.conversationHistory.push({
        role: 'assistant',
        content: JSON.stringify(response.content),
      });

      // Execute each tool
      const toolResults: any[] = [];

      for (const toolUse of toolUseBlocks) {
        if (toolUse.type !== 'tool_use') continue;

        this.trace('TOOL_CALL', {
          toolName: toolUse.name,
          toolInput: toolUse.input,
          toolUseId: toolUse.id,
          toolRound,
        });

        try {
          // Check if tool is still enabled before executing
          if (!isToolEnabled(toolUse.name)) {
            throw new Error(`Tool '${toolUse.name}' is currently disabled`);
          }

          const result = executeTool(toolUse.name, toolUse.input);

          this.trace('TOOL_RESULT', {
            toolName: toolUse.name,
            toolUseId: toolUse.id,
            result,
            success: true,
            toolRound,
          });

          toolResults.push({
            type: 'tool_result',
            tool_use_id: toolUse.id,
            content: JSON.stringify(result),
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';

          this.trace('TOOL_RESULT', {
            toolName: toolUse.name,
            toolUseId: toolUse.id,
            error: errorMessage,
            success: false,
            toolRound,
          });

          toolResults.push({
            type: 'tool_result',
            tool_use_id: toolUse.id,
            content: JSON.stringify({ error: errorMessage }),
            is_error: true,
          });
        }
      }

      // Add tool results to conversation as user message
      this.conversationHistory.push({
        role: 'user',
        content: JSON.stringify(toolResults),
      });

      toolRound++;
    }

    if (toolRound >= this.maxToolRounds) {
      finalResponse = 'Maximum tool rounds reached. Unable to complete the request.';
      this.trace('MAX_TOOL_ROUNDS_REACHED', {
        maxToolRounds: this.maxToolRounds,
      });
    }

    return finalResponse;
  }
}
