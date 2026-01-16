/**
 * Read Tool
 * Read any file in the working directory
 */
export interface ReadInput {
    file_path: string;
    offset?: number;
    limit?: number;
}
export declare function read(input: ReadInput): {
    content: string;
    lines_read: number;
};
export declare const readToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            file_path: {
                type: string;
                description: string;
            };
            offset: {
                type: string;
                description: string;
            };
            limit: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=read.d.ts.map