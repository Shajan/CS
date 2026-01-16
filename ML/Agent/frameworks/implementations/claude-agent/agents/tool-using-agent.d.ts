import { BaseAgent, AgentConfig } from './base-agent.js';
/**
 * ToolUsingAgent - An agent that can use tools to perform actions
 *
 * This agent extends BaseAgent and adds the ability to:
 * - Call tools based on user requests
 * - Execute tools and return results to Claude
 * - Handle multi-turn tool interactions
 */
export declare class ToolUsingAgent extends BaseAgent {
    private systemPrompt;
    private maxToolRounds;
    constructor(config: AgentConfig, systemPrompt?: string, maxToolRounds?: number);
    /**
     * Chat with the agent using tools when needed
     */
    chat(userMessage: string): Promise<string>;
}
//# sourceMappingURL=tool-using-agent.d.ts.map