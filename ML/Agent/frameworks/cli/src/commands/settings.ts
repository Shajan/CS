import chalk from 'chalk';
import {
  getFrameworkSettings,
  updateFrameworkSettings,
  getMCPServers,
  addMCPServer,
  updateMCPServer,
  deleteMCPServer,
  type MCPServer,
} from '../api/client.js';
import { error, success, header, table } from '../utils/output.js';

interface FrameworkGetOptions {
  json?: boolean;
}

interface FrameworkSetOptions {
  tools?: string;
  json?: boolean;
}

interface MCPListOptions {
  json?: boolean;
}

interface MCPAddOptions {
  command: string;
  args?: string;
  env?: string;
  disabled?: boolean;
}

/**
 * Get framework settings
 */
export async function settingsFrameworkGet(
  framework: string,
  options: FrameworkGetOptions
): Promise<void> {
  try {
    const settings = await getFrameworkSettings(framework);

    if (options.json) {
      console.log(JSON.stringify(settings, null, 2));
      return;
    }

    header(`Settings for ${framework}`);

    if (!settings || Object.keys(settings).length === 0) {
      console.log(chalk.yellow('No settings configured'));
      return;
    }

    if (settings.enabledTools && Array.isArray(settings.enabledTools)) {
      console.log(chalk.bold('Enabled Tools:'));
      if (settings.enabledTools.length === 0) {
        console.log(chalk.dim('  (none)'));
      } else {
        settings.enabledTools.forEach((tool: string) => {
          console.log(chalk.cyan(`  • ${tool}`));
        });
      }
      console.log();
    }

    if (settings.customConfig) {
      console.log(chalk.bold('Custom Config:'));
      console.log(chalk.dim(JSON.stringify(settings.customConfig, null, 2)));
      console.log();
    }
  } catch (err) {
    error(`Failed to get settings: ${err instanceof Error ? err.message : 'Unknown error'}`);
    process.exit(1);
  }
}

/**
 * Set framework settings
 */
export async function settingsFrameworkSet(
  framework: string,
  options: FrameworkSetOptions
): Promise<void> {
  try {
    const settings = await getFrameworkSettings(framework);

    if (options.tools !== undefined) {
      const tools = options.tools.split(',').map((t) => t.trim()).filter((t) => t.length > 0);
      settings.enabledTools = tools;
    }

    await updateFrameworkSettings(framework, settings);

    if (options.json) {
      console.log(JSON.stringify({ success: true, settings }, null, 2));
      return;
    }

    success(`Updated settings for ${framework}`);
    if (options.tools !== undefined) {
      console.log(chalk.dim(`Enabled tools: ${settings.enabledTools.join(', ') || '(none)'}`));
    }
  } catch (err) {
    error(`Failed to update settings: ${err instanceof Error ? err.message : 'Unknown error'}`);
    process.exit(1);
  }
}

/**
 * List MCP servers
 */
export async function settingsMCPList(options: MCPListOptions): Promise<void> {
  try {
    const servers = await getMCPServers();

    if (options.json) {
      console.log(JSON.stringify(servers, null, 2));
      return;
    }

    header(`MCP Servers (${servers.length})`);

    if (servers.length === 0) {
      console.log(chalk.yellow('No MCP servers configured'));
      return;
    }

    const tableData = servers.map((server) => [
      server.name,
      server.enabled ? chalk.green('✓') : chalk.red('✗'),
      server.command,
      server.args ? server.args.join(' ') : '',
    ]);

    table(['Name', 'Enabled', 'Command', 'Args'], tableData, [20, 8, 40]);
  } catch (err) {
    error(`Failed to list MCP servers: ${err instanceof Error ? err.message : 'Unknown error'}`);
    process.exit(1);
  }
}

/**
 * Add MCP server
 */
export async function settingsMCPAdd(name: string, options: MCPAddOptions): Promise<void> {
  try {
    const server: MCPServer = {
      name,
      command: options.command,
      args: options.args ? options.args.split(' ').filter((a) => a.length > 0) : undefined,
      env: options.env
        ? JSON.parse(options.env)
        : undefined,
      enabled: !options.disabled,
    };

    await addMCPServer(server);
    success(`Added MCP server: ${name}`);
  } catch (err) {
    error(`Failed to add MCP server: ${err instanceof Error ? err.message : 'Unknown error'}`);
    process.exit(1);
  }
}

/**
 * Remove MCP server
 */
export async function settingsMCPRemove(name: string): Promise<void> {
  try {
    await deleteMCPServer(name);
    success(`Removed MCP server: ${name}`);
  } catch (err) {
    error(`Failed to remove MCP server: ${err instanceof Error ? err.message : 'Unknown error'}`);
    process.exit(1);
  }
}

/**
 * Enable/disable MCP server
 */
export async function settingsMCPToggle(
  name: string,
  enable: boolean
): Promise<void> {
  try {
    await updateMCPServer(name, { enabled: enable });
    success(`${enable ? 'Enabled' : 'Disabled'} MCP server: ${name}`);
  } catch (err) {
    error(
      `Failed to ${enable ? 'enable' : 'disable'} MCP server: ${err instanceof Error ? err.message : 'Unknown error'}`
    );
    process.exit(1);
  }
}
