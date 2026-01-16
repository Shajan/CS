/**
 * Grep Tool
 * Search file contents using regex patterns
 */
export interface GrepInput {
    pattern: string;
    path?: string;
    glob?: string;
    case_insensitive?: boolean;
    output_mode?: 'content' | 'files_with_matches' | 'count';
    context_lines?: number;
}
export declare function grep(input: GrepInput): {
    results: string;
    matches: number;
};
export declare const grepToolDefinition: {
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
            glob: {
                type: string;
                description: string;
            };
            case_insensitive: {
                type: string;
                description: string;
            };
            output_mode: {
                type: string;
                enum: string[];
                description: string;
            };
            context_lines: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=grep.d.ts.map