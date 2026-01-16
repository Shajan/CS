/**
 * Tool Registry
 * Central place to register and manage agent tools
 */
import { calculator, calculatorToolDefinition } from './calculator.js';
import { getTime, getTimeToolDefinition } from './get-time.js';
import { read, readToolDefinition } from './read.js';
import { write, writeToolDefinition } from './write.js';
import { edit, editToolDefinition } from './edit.js';
import { bash, bashToolDefinition } from './bash.js';
import { globTool, globToolDefinition } from './glob.js';
import { grep, grepToolDefinition } from './grep.js';
import { webSearch, webSearchToolDefinition } from './web-search.js';
import { webFetch, webFetchToolDefinition } from './web-fetch.js';
import { askUserQuestion, askUserQuestionToolDefinition } from './ask-user-question.js';
export const tools = {
    calculator: {
        definition: calculatorToolDefinition,
        execute: (input) => calculator(input),
    },
    get_time: {
        definition: getTimeToolDefinition,
        execute: (input) => getTime(input),
    },
    read: {
        definition: readToolDefinition,
        execute: (input) => read(input),
    },
    write: {
        definition: writeToolDefinition,
        execute: (input) => write(input),
    },
    edit: {
        definition: editToolDefinition,
        execute: (input) => edit(input),
    },
    bash: {
        definition: bashToolDefinition,
        execute: (input) => bash(input),
    },
    glob: {
        definition: globToolDefinition,
        execute: (input) => globTool(input),
    },
    grep: {
        definition: grepToolDefinition,
        execute: (input) => grep(input),
    },
    web_search: {
        definition: webSearchToolDefinition,
        execute: (input) => webSearch(input),
    },
    web_fetch: {
        definition: webFetchToolDefinition,
        execute: (input) => webFetch(input),
    },
    ask_user_question: {
        definition: askUserQuestionToolDefinition,
        execute: (input) => askUserQuestion(input),
    },
};
export const toolDefinitions = Object.values(tools).map((tool) => tool.definition);
export function executeTool(toolName, input) {
    const tool = tools[toolName];
    if (!tool) {
        throw new Error(`Unknown tool: ${toolName}`);
    }
    return tool.execute(input);
}
// Tool enablement (for now, all tools are enabled by default)
const enabledTools = new Set(Object.keys(tools));
export function getEnabledTools() {
    return Array.from(enabledTools)
        .map((toolName) => tools[toolName]?.definition)
        .filter(Boolean);
}
export function isToolEnabled(toolName) {
    return enabledTools.has(toolName);
}
export function enableTool(toolName) {
    if (tools[toolName]) {
        enabledTools.add(toolName);
    }
}
export function disableTool(toolName) {
    enabledTools.delete(toolName);
}
//# sourceMappingURL=index.js.map