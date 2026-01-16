/**
 * Bash Tool
 * Execute bash commands and scripts
 */
export interface BashInput {
    command: string;
    timeout?: number;
}
export declare function bash(input: BashInput): {
    stdout: string;
    stderr: string;
    exitCode: number;
};
export declare const bashToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            command: {
                type: string;
                description: string;
            };
            timeout: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=bash.d.ts.map