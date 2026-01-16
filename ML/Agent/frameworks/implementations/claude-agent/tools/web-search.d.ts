/**
 * WebSearch Tool
 * Search the web for current information
 */
export interface WebSearchInput {
    query: string;
    num_results?: number;
}
export declare function webSearch(input: WebSearchInput): {
    results: any[];
    query: string;
};
export declare const webSearchToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            query: {
                type: string;
                description: string;
            };
            num_results: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=web-search.d.ts.map