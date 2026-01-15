/**
 * Grep Tool
 * Search file contents using regex patterns
 */

import { execSync } from 'child_process';
import { resolve } from 'path';

export interface GrepInput {
  pattern: string;
  path?: string;
  glob?: string;
  case_insensitive?: boolean;
  output_mode?: 'content' | 'files_with_matches' | 'count';
  context_lines?: number;
}

export function grep(input: GrepInput): { results: string; matches: number } {
  const {
    pattern,
    path = process.cwd(),
    glob,
    case_insensitive = false,
    output_mode = 'files_with_matches',
    context_lines = 0,
  } = input;

  try {
    const searchPath = resolve(process.cwd(), path);

    // Build rg (ripgrep) command
    let command = 'rg';

    if (case_insensitive) {
      command += ' -i';
    }

    if (output_mode === 'files_with_matches') {
      command += ' -l';
    } else if (output_mode === 'count') {
      command += ' -c';
    }

    if (context_lines > 0 && output_mode === 'content') {
      command += ` -C ${context_lines}`;
    }

    if (glob) {
      command += ` --glob "${glob}"`;
    }

    command += ` "${pattern}" "${searchPath}"`;

    try {
      const results = execSync(command, {
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024,
      });

      const lines = results.trim().split('\n');

      return {
        results: results.trim(),
        matches: lines.length,
      };
    } catch (error: any) {
      // rg returns exit code 1 when no matches found
      if (error.status === 1) {
        return {
          results: '',
          matches: 0,
        };
      }
      throw error;
    }
  } catch (error) {
    throw new Error(`Grep search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export const grepToolDefinition = {
  name: 'grep',
  description: 'Powerful search tool for finding text in files using regex patterns. Supports filtering by file type and various output modes.',
  input_schema: {
    type: 'object',
    properties: {
      pattern: {
        type: 'string',
        description: 'The regex pattern to search for',
      },
      path: {
        type: 'string',
        description: 'File or directory to search in (optional)',
      },
      glob: {
        type: 'string',
        description: 'Glob pattern to filter files (e.g., "*.js", "**/*.ts")',
      },
      case_insensitive: {
        type: 'boolean',
        description: 'Case insensitive search (default: false)',
      },
      output_mode: {
        type: 'string',
        enum: ['content', 'files_with_matches', 'count'],
        description: 'Output mode: "content" (matching lines), "files_with_matches" (file paths), "count" (match counts)',
      },
      context_lines: {
        type: 'number',
        description: 'Number of context lines to show around matches (only for content mode)',
      },
    },
    required: ['pattern'],
  },
};
