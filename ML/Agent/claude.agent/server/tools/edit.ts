/**
 * Edit Tool
 * Make precise edits to existing files by replacing old_string with new_string
 */

import { readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

export interface EditInput {
  file_path: string;
  old_string: string;
  new_string: string;
  replace_all?: boolean;
}

export function edit(input: EditInput): { success: boolean; replacements: number; message: string } {
  const { file_path, old_string, new_string, replace_all = false } = input;

  try {
    const absolutePath = resolve(process.cwd(), file_path);
    const content = readFileSync(absolutePath, 'utf-8');

    if (!content.includes(old_string)) {
      throw new Error(`String not found in file: "${old_string}"`);
    }

    let newContent: string;
    let replacements: number;

    if (replace_all) {
      const parts = content.split(old_string);
      replacements = parts.length - 1;
      newContent = parts.join(new_string);
    } else {
      // Replace only first occurrence
      const occurrences = content.split(old_string).length - 1;
      if (occurrences > 1) {
        throw new Error(`String appears ${occurrences} times in file. Use replace_all: true to replace all occurrences.`);
      }
      replacements = 1;
      newContent = content.replace(old_string, new_string);
    }

    writeFileSync(absolutePath, newContent, 'utf-8');

    return {
      success: true,
      replacements,
      message: `Successfully replaced ${replacements} occurrence(s) in ${file_path}`,
    };
  } catch (error) {
    throw new Error(`Failed to edit file: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export const editToolDefinition = {
  name: 'edit',
  description: 'Performs exact string replacements in files. The old_string must be unique in the file unless replace_all is true.',
  input_schema: {
    type: 'object',
    properties: {
      file_path: {
        type: 'string',
        description: 'The path to the file to modify',
      },
      old_string: {
        type: 'string',
        description: 'The text to replace',
      },
      new_string: {
        type: 'string',
        description: 'The text to replace it with',
      },
      replace_all: {
        type: 'boolean',
        description: 'Replace all occurrences (default: false)',
      },
    },
    required: ['file_path', 'old_string', 'new_string'],
  },
};
