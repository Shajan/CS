import readline from 'readline/promises';
import chalk from 'chalk';
import { sendChatMessage, clearHistory } from '../api/client.js';
import { error, success, formatMetadata } from '../utils/output.js';

interface ChatOptions {
  framework?: string;
  session?: string;
}

/**
 * Interactive chat command
 */
export async function chatCommand(options: ChatOptions): Promise<void> {
  const framework = options.framework || 'claude-agent';
  const sessionId = options.session || `session-${Date.now()}`;

  console.log(chalk.bold.cyan('\n🤖 Multi-Framework Agent Chat\n'));
  console.log(chalk.dim(`Framework: ${framework}`));
  console.log(chalk.dim(`Session: ${sessionId}`));
  console.log(chalk.dim(`\nCommands: 'exit' to quit, 'clear' to clear history\n`));

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    while (true) {
      const input = await rl.question(chalk.blue('You: '));

      if (!input.trim()) continue;

      // Handle special commands
      if (input.toLowerCase() === 'exit') {
        console.log(chalk.dim('\nGoodbye! 👋\n'));
        break;
      }

      if (input.toLowerCase() === 'clear') {
        try {
          await clearHistory(sessionId, framework);
          success('History cleared');
        } catch (err) {
          error(`Failed to clear history: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
        continue;
      }

      // Send message
      try {
        const startTime = Date.now();
        const response = await sendChatMessage({
          message: input,
          sessionId,
          framework,
        });
        const duration = Date.now() - startTime;

        console.log(chalk.green('\nAssistant:'), response.response);

        // Print metadata
        const metadata = formatMetadata({
          ...response.metadata,
          duration,
        });
        if (metadata) {
          console.log(chalk.dim(metadata));
        }
        console.log();
      } catch (err) {
        error(
          `Failed to send message: ${err instanceof Error ? err.message : 'Unknown error'}`
        );
      }
    }
  } finally {
    rl.close();
  }
}
