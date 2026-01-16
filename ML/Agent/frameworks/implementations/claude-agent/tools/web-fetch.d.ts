/**
 * WebFetch Tool
 * Fetch and parse web page content
 */
export interface WebFetchInput {
    url: string;
    prompt: string;
}
export declare function webFetch(input: WebFetchInput): Promise<{
    content: string;
    url: string;
}>;
export declare const webFetchToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            url: {
                type: string;
                description: string;
            };
            prompt: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=web-fetch.d.ts.map