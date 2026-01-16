import chalk from 'chalk';
import { sendChatMessage, listFrameworks } from '../api/client.js';
import { error, success, header, table } from '../utils/output.js';
import * as fs from 'fs';

interface BenchmarkOptions {
  message: string;
  frameworks?: string;
  output?: string;
  json?: boolean;
}

interface BenchmarkResult {
  framework: string;
  success: boolean;
  response?: string;
  duration?: number;
  tokensUsed?: number;
  error?: string;
}

/**
 * Benchmark command - test all frameworks with same message
 */
export async function benchmarkCommand(options: BenchmarkOptions): Promise<void> {
  if (!options.message) {
    error('Message is required. Use -m or --message option');
    process.exit(1);
  }

  try {
    // Get frameworks to test
    let frameworksToTest: string[];
    if (options.frameworks) {
      frameworksToTest = options.frameworks.split(',').map((f) => f.trim());
    } else {
      const allFrameworks = await listFrameworks();
      frameworksToTest = allFrameworks.map((f) => f.name);
    }

    if (!options.json) {
      header('Framework Benchmark');
      console.log(chalk.dim(`Testing ${frameworksToTest.length} framework(s)`));
      console.log(chalk.dim(`Message: "${options.message}"\n`));
    }

    const results: BenchmarkResult[] = [];

    // Test each framework
    for (const framework of frameworksToTest) {
      if (!options.json) {
        process.stdout.write(chalk.dim(`Testing ${framework}... `));
      }

      try {
        const sessionId = `benchmark-${Date.now()}-${framework}`;
        const startTime = Date.now();

        const response = await sendChatMessage({
          message: options.message,
          sessionId,
          framework,
        });

        const duration = Date.now() - startTime;

        results.push({
          framework,
          success: true,
          response: response.response,
          duration,
          tokensUsed: response.metadata?.tokensUsed,
        });

        if (!options.json) {
          console.log(chalk.green(`✓ ${duration}ms`));
        }
      } catch (err) {
        results.push({
          framework,
          success: false,
          error: err instanceof Error ? err.message : 'Unknown error',
        });

        if (!options.json) {
          console.log(chalk.red(`✗ Error`));
        }
      }
    }

    // Output results
    if (options.json) {
      console.log(JSON.stringify(results, null, 2));
    } else {
      console.log();
      header('Results');

      const tableData = results.map((result) => [
        result.framework,
        result.success ? chalk.green('✓') : chalk.red('✗'),
        result.duration ? `${result.duration}ms` : '-',
        result.tokensUsed ? result.tokensUsed.toString() : '-',
        result.error ? chalk.red(result.error) : '',
      ]);

      table(
        ['Framework', 'Status', 'Duration', 'Tokens', 'Error'],
        tableData
      );

      // Summary
      const successCount = results.filter((r) => r.success).length;
      const avgDuration =
        results
          .filter((r) => r.success && r.duration)
          .reduce((sum, r) => sum + (r.duration || 0), 0) / successCount || 0;

      console.log(chalk.bold('Summary:'));
      console.log(chalk.dim(`  Successful: ${successCount}/${results.length}`));
      console.log(chalk.dim(`  Average duration: ${avgDuration.toFixed(0)}ms`));
      console.log();
    }

    // Save to file if requested
    if (options.output) {
      const timestamp = new Date().toISOString();
      const data = {
        timestamp,
        message: options.message,
        results,
      };
      fs.writeFileSync(options.output, JSON.stringify(data, null, 2));
      success(`Results saved to ${options.output}`);
    }
  } catch (err) {
    error(`Benchmark failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    process.exit(1);
  }
}
