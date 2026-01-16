/**
 * Bash Tool
 * Execute bash commands and scripts
 */
import { execSync } from 'child_process';
export function bash(input) {
    const { command, timeout = 30000 } = input;
    try {
        const stdout = execSync(command, {
            encoding: 'utf-8',
            timeout,
            maxBuffer: 10 * 1024 * 1024, // 10MB
            cwd: process.cwd(),
        });
        return {
            stdout: stdout || '',
            stderr: '',
            exitCode: 0,
        };
    }
    catch (error) {
        return {
            stdout: error.stdout || '',
            stderr: error.stderr || error.message || 'Command execution failed',
            exitCode: error.status || 1,
        };
    }
}
export const bashToolDefinition = {
    name: 'bash',
    description: 'Executes a bash command in a shell. Returns stdout, stderr, and exit code. Use for git operations, file operations, running scripts, etc.',
    input_schema: {
        type: 'object',
        properties: {
            command: {
                type: 'string',
                description: 'The bash command to execute',
            },
            timeout: {
                type: 'number',
                description: 'Timeout in milliseconds (default: 30000)',
            },
        },
        required: ['command'],
    },
};
//# sourceMappingURL=bash.js.map