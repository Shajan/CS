/**
 * Glob Tool
 * Find files by pattern matching
 */
export interface GlobInput {
    pattern: string;
    path?: string;
}
export declare function globTool(input: GlobInput): {
    files: string[];
    count: number;
};
export declare const globToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            pattern: {
                type: string;
                description: string;
            };
            path: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=glob.d.ts.map