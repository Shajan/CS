import chalk from 'chalk';
import { getFrameworkInfo } from '../api/client.js';
import { error, header, success } from '../utils/output.js';

interface InfoOptions {
  json?: boolean;
}

/**
 * Info command - show details about a specific framework
 */
export async function infoCommand(framework: string, options: InfoOptions): Promise<void> {
  if (!framework) {
    error('Framework name is required');
    process.exit(1);
  }

  try {
    const info = await getFrameworkInfo(framework);

    if (options.json) {
      console.log(JSON.stringify(info, null, 2));
      return;
    }

    header(info.displayName);

    console.log(chalk.bold('Basic Information:'));
    console.log(chalk.dim(`  Name: ${info.name}`));
    console.log(chalk.dim(`  Version: ${info.version}`));
    if (info.description) {
      console.log(chalk.dim(`  Description: ${info.description}`));
    }
    console.log();

    if (info.capabilities) {
      console.log(chalk.bold('Capabilities:'));
      console.log(
        chalk.dim(
          `  Streaming: ${info.capabilities.supportsStreaming ? '✓' : '✗'}`
        )
      );
      console.log(
        chalk.dim(`  Tools: ${info.capabilities.supportsTools ? '✓' : '✗'}`)
      );
      console.log(
        chalk.dim(
          `  Multi-Modal: ${info.capabilities.supportsMultiModal ? '✓' : '✗'}`
        )
      );
      console.log(
        chalk.dim(
          `  Multi-Agent: ${info.capabilities.supportsMultiAgent ? '✓' : '✗'}`
        )
      );
      console.log(
        chalk.dim(`  Memory: ${info.capabilities.supportsMemory ? '✓' : '✗'}`)
      );

      if (info.capabilities.maxContextLength) {
        console.log(
          chalk.dim(
            `  Max Context: ${info.capabilities.maxContextLength.toLocaleString()} tokens`
          )
        );
      }

      if (info.capabilities.supportedModels) {
        console.log(chalk.dim(`  Supported Models:`));
        info.capabilities.supportedModels.forEach((model) => {
          console.log(chalk.dim(`    - ${model}`));
        });
      }
      console.log();
    }

    success('Framework information retrieved');
  } catch (err) {
    error(
      `Failed to get framework info: ${err instanceof Error ? err.message : 'Unknown error'}`
    );
    process.exit(1);
  }
}
