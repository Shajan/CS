/**
 * Tool Registry
 * Central place to register and manage agent tools
 */

import { calculator, calculatorToolDefinition, CalculatorInput } from './calculator.js';
import { getTime, getTimeToolDefinition, GetTimeInput } from './get-time.js';
import { read, readToolDefinition, ReadInput } from './read.js';
import { write, writeToolDefinition, WriteInput } from './write.js';
import { edit, editToolDefinition, EditInput } from './edit.js';
import { bash, bashToolDefinition, BashInput } from './bash.js';
import { globTool, globToolDefinition, GlobInput } from './glob.js';
import { grep, grepToolDefinition, GrepInput } from './grep.js';
import { webSearch, webSearchToolDefinition, WebSearchInput } from './web-search.js';
import { webFetch, webFetchToolDefinition, WebFetchInput } from './web-fetch.js';
import { askUserQuestion, askUserQuestionToolDefinition, AskUserQuestionInput } from './ask-user-question.js';

export interface Tool {
  definition: any;
  execute: (input: any) => any;
}

export const tools: Record<string, Tool> = {
  calculator: {
    definition: calculatorToolDefinition,
    execute: (input: CalculatorInput) => calculator(input),
  },
  get_time: {
    definition: getTimeToolDefinition,
    execute: (input: GetTimeInput) => getTime(input),
  },
  read: {
    definition: readToolDefinition,
    execute: (input: ReadInput) => read(input),
  },
  write: {
    definition: writeToolDefinition,
    execute: (input: WriteInput) => write(input),
  },
  edit: {
    definition: editToolDefinition,
    execute: (input: EditInput) => edit(input),
  },
  bash: {
    definition: bashToolDefinition,
    execute: (input: BashInput) => bash(input),
  },
  glob: {
    definition: globToolDefinition,
    execute: (input: GlobInput) => globTool(input),
  },
  grep: {
    definition: grepToolDefinition,
    execute: (input: GrepInput) => grep(input),
  },
  web_search: {
    definition: webSearchToolDefinition,
    execute: (input: WebSearchInput) => webSearch(input),
  },
  web_fetch: {
    definition: webFetchToolDefinition,
    execute: (input: WebFetchInput) => webFetch(input),
  },
  ask_user_question: {
    definition: askUserQuestionToolDefinition,
    execute: (input: AskUserQuestionInput) => askUserQuestion(input),
  },
};

export const toolDefinitions = Object.values(tools).map((tool) => tool.definition);

export function executeTool(toolName: string, input: any): any {
  const tool = tools[toolName];
  if (!tool) {
    throw new Error(`Unknown tool: ${toolName}`);
  }
  return tool.execute(input);
}

// Tool enablement (for now, all tools are enabled by default)
const enabledTools = new Set(Object.keys(tools));

export function getEnabledTools(): any[] {
  return Array.from(enabledTools)
    .map((toolName) => tools[toolName]?.definition)
    .filter(Boolean);
}

export function isToolEnabled(toolName: string): boolean {
  return enabledTools.has(toolName);
}

export function enableTool(toolName: string): void {
  if (tools[toolName]) {
    enabledTools.add(toolName);
  }
}

export function disableTool(toolName: string): void {
  enabledTools.delete(toolName);
}
