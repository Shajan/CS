import chalk from 'chalk';
import { sendChatMessage } from '../api/client.js';
import { error, success, header } from '../utils/output.js';

interface TestOptions {
  framework: string;
  message: string;
  session?: string;
  json?: boolean;
}

/**
 * Test command - send a single message
 */
export async function testCommand(options: TestOptions): Promise<void> {
  if (!options.framework) {
    error('Framework is required. Use -f or --framework option');
    process.exit(1);
  }

  if (!options.message) {
    error('Message is required. Use -m or --message option');
    process.exit(1);
  }

  const sessionId = options.session || `test-${Date.now()}`;

  try {
    if (!options.json) {
      header(`Testing ${options.framework}`);
      console.log(chalk.dim(`Message: ${options.message}\n`));
    }

    const startTime = Date.now();
    const response = await sendChatMessage({
      message: options.message,
      sessionId,
      framework: options.framework,
    });
    const duration = Date.now() - startTime;

    if (options.json) {
      // Output as JSON
      console.log(
        JSON.stringify(
          {
            ...response,
            metadata: {
              ...response.metadata,
              duration,
            },
          },
          null,
          2
        )
      );
    } else {
      // Formatted output
      console.log(chalk.bold('Response:'));
      console.log(response.response);
      console.log();

      console.log(chalk.bold('Metadata:'));
      console.log(chalk.dim(`  Framework: ${response.framework}`));
      if (response.metadata?.model) {
        console.log(chalk.dim(`  Model: ${response.metadata.model}`));
      }
      if (response.metadata?.tokensUsed) {
        console.log(chalk.dim(`  Tokens: ${response.metadata.tokensUsed}`));
      }
      console.log(chalk.dim(`  Duration: ${duration}ms`));
      console.log();

      success('Test completed');
    }
  } catch (err) {
    if (options.json) {
      console.error(
        JSON.stringify(
          {
            error: err instanceof Error ? err.message : 'Unknown error',
          },
          null,
          2
        )
      );
    } else {
      error(`Test failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
    process.exit(1);
  }
}
