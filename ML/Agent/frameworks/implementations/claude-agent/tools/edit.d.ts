/**
 * Edit Tool
 * Make precise edits to existing files by replacing old_string with new_string
 */
export interface EditInput {
    file_path: string;
    old_string: string;
    new_string: string;
    replace_all?: boolean;
}
export declare function edit(input: EditInput): {
    success: boolean;
    replacements: number;
    message: string;
};
export declare const editToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            file_path: {
                type: string;
                description: string;
            };
            old_string: {
                type: string;
                description: string;
            };
            new_string: {
                type: string;
                description: string;
            };
            replace_all: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=edit.d.ts.map