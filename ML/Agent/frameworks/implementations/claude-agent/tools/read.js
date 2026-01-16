/**
 * Read Tool
 * Read any file in the working directory
 */
import { readFileSync } from 'fs';
import { resolve } from 'path';
export function read(input) {
    const { file_path, offset = 0, limit } = input;
    try {
        const absolutePath = resolve(process.cwd(), file_path);
        const content = readFileSync(absolutePath, 'utf-8');
        const lines = content.split('\n');
        const startLine = offset;
        const endLine = limit ? offset + limit : lines.length;
        const selectedLines = lines.slice(startLine, endLine);
        // Format with line numbers like cat -n
        const numberedLines = selectedLines.map((line, idx) => `${String(startLine + idx + 1).padStart(6)}→${line}`).join('\n');
        return {
            content: numberedLines,
            lines_read: selectedLines.length,
        };
    }
    catch (error) {
        throw new Error(`Failed to read file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}
export const readToolDefinition = {
    name: 'read',
    description: 'Reads a file from the local filesystem. Returns file contents with line numbers.',
    input_schema: {
        type: 'object',
        properties: {
            file_path: {
                type: 'string',
                description: 'The path to the file to read (relative or absolute)',
            },
            offset: {
                type: 'number',
                description: 'The line number to start reading from (optional)',
            },
            limit: {
                type: 'number',
                description: 'The number of lines to read (optional)',
            },
        },
        required: ['file_path'],
    },
};
//# sourceMappingURL=read.js.map