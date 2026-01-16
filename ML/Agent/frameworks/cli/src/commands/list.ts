import chalk from 'chalk';
import { listFrameworks } from '../api/client.js';
import { error, header, table } from '../utils/output.js';

interface ListOptions {
  json?: boolean;
}

/**
 * List command - show all available frameworks
 */
export async function listCommand(options: ListOptions): Promise<void> {
  try {
    const frameworks = await listFrameworks();

    if (options.json) {
      console.log(JSON.stringify(frameworks, null, 2));
      return;
    }

    header('Available Frameworks');

    if (frameworks.length === 0) {
      console.log(chalk.yellow('No frameworks available'));
      return;
    }

    const tableData = frameworks.map((framework) => {
      const badges: string[] = [];
      if (framework.capabilities?.supportsTools) badges.push('🔧');
      if (framework.capabilities?.supportsStreaming) badges.push('📡');
      if (framework.capabilities?.supportsMultiModal) badges.push('🖼️');
      if (framework.capabilities?.supportsMultiAgent) badges.push('👥');

      return [
        framework.name,
        framework.displayName,
        framework.version,
        badges.join(' '),
        framework.description || '',
      ];
    });

    table(
      ['Name', 'Display Name', 'Version', 'Features', 'Description'],
      tableData,
      [20, 25, 10, 12]
    );

    console.log(chalk.dim('Features: 🔧 Tools | 📡 Streaming | 🖼️ Multi-Modal | 👥 Multi-Agent\n'));
  } catch (err) {
    error(`Failed to list frameworks: ${err instanceof Error ? err.message : 'Unknown error'}`);
    process.exit(1);
  }
}
