/**
 * Glob Tool
 * Find files by pattern matching
 */

import { globSync } from 'glob';
import { resolve } from 'path';
import { statSync } from 'fs';

export interface GlobInput {
  pattern: string;
  path?: string;
}

export function globTool(input: GlobInput): { files: string[]; count: number } {
  const { pattern, path = process.cwd() } = input;

  try {
    const searchPath = resolve(process.cwd(), path);

    const files = globSync(pattern, {
      cwd: searchPath,
      absolute: false,
      nodir: true,
      ignore: ['**/node_modules/**', '**/.git/**'],
    });

    // Sort by modification time (most recent first)
    const sortedFiles = files.sort((a, b) => {
      try {
        const statA = statSync(resolve(searchPath, a));
        const statB = statSync(resolve(searchPath, b));
        return statB.mtimeMs - statA.mtimeMs;
      } catch {
        return 0;
      }
    });

    return {
      files: sortedFiles,
      count: sortedFiles.length,
    };
  } catch (error) {
    throw new Error(`Glob search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export const globToolDefinition = {
  name: 'glob',
  description: 'Fast file pattern matching tool. Supports glob patterns like "**/*.js" or "src/**/*.ts". Returns matching file paths sorted by modification time.',
  input_schema: {
    type: 'object',
    properties: {
      pattern: {
        type: 'string',
        description: 'The glob pattern to match files against (e.g., "**/*.ts", "src/**/*.js")',
      },
      path: {
        type: 'string',
        description: 'The directory to search in (optional, defaults to current working directory)',
      },
    },
    required: ['pattern'],
  },
};
