/**
 * Write Tool
 * Create new files or overwrite existing files
 */
export interface WriteInput {
    file_path: string;
    content: string;
}
export declare function write(input: WriteInput): {
    success: boolean;
    message: string;
};
export declare const writeToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            file_path: {
                type: string;
                description: string;
            };
            content: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=write.d.ts.map