/**
 * Tool Registry
 * Central place to register and manage agent tools
 */
export interface Tool {
    definition: any;
    execute: (input: any) => any;
}
export declare const tools: Record<string, Tool>;
export declare const toolDefinitions: any[];
export declare function executeTool(toolName: string, input: any): any;
export declare function getEnabledTools(): any[];
export declare function isToolEnabled(toolName: string): boolean;
export declare function enableTool(toolName: string): void;
export declare function disableTool(toolName: string): void;
//# sourceMappingURL=index.d.ts.map