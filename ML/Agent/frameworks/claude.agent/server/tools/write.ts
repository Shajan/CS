/**
 * Write Tool
 * Create new files or overwrite existing files
 */

import { writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { mkdirSync } from 'fs';

export interface WriteInput {
  file_path: string;
  content: string;
}

export function write(input: WriteInput): { success: boolean; message: string } {
  const { file_path, content } = input;

  try {
    const absolutePath = resolve(process.cwd(), file_path);

    // Create directory if it doesn't exist
    const dir = dirname(absolutePath);
    mkdirSync(dir, { recursive: true });

    writeFileSync(absolutePath, content, 'utf-8');

    return {
      success: true,
      message: `Successfully wrote ${content.length} characters to ${file_path}`,
    };
  } catch (error) {
    throw new Error(`Failed to write file: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export const writeToolDefinition = {
  name: 'write',
  description: 'Writes a file to the local filesystem. Creates directories if needed. Overwrites existing files.',
  input_schema: {
    type: 'object',
    properties: {
      file_path: {
        type: 'string',
        description: 'The path where to write the file',
      },
      content: {
        type: 'string',
        description: 'The content to write to the file',
      },
    },
    required: ['file_path', 'content'],
  },
};
