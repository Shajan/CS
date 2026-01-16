#!/usr/bin/env node

import { Command } from 'commander';
import { chatCommand } from './commands/chat.js';
import { testCommand } from './commands/test.js';
import { benchmarkCommand } from './commands/benchmark.js';
import { listCommand } from './commands/list.js';
import { infoCommand } from './commands/info.js';
import {
  settingsFrameworkGet,
  settingsFrameworkSet,
  settingsMCPList,
  settingsMCPAdd,
  settingsMCPRemove,
  settingsMCPToggle,
} from './commands/settings.js';

const program = new Command();

program
  .name('agent-cli')
  .description('CLI for multi-framework AI agent system')
  .version('1.0.0');

// Chat command
program
  .command('chat')
  .description('Start an interactive chat session')
  .option('-f, --framework <name>', 'Framework to use', 'claude-agent')
  .option('-s, --session <id>', 'Session ID')
  .action(chatCommand);

// Test command
program
  .command('test')
  .description('Test a framework with a single message')
  .requiredOption('-f, --framework <name>', 'Framework to use')
  .requiredOption('-m, --message <text>', 'Message to send')
  .option('-s, --session <id>', 'Session ID')
  .option('--json', 'Output as JSON')
  .action(testCommand);

// Benchmark command
program
  .command('benchmark')
  .description('Compare frameworks with the same message')
  .requiredOption('-m, --message <text>', 'Message to send')
  .option('-f, --frameworks <list>', 'Comma-separated frameworks (default: all)')
  .option('--output <file>', 'Save results to file')
  .option('--json', 'Output as JSON')
  .action(benchmarkCommand);

// List command
program
  .command('list')
  .description('List all available frameworks')
  .option('--json', 'Output as JSON')
  .action(listCommand);

// Info command
program
  .command('info <framework>')
  .description('Get details about a specific framework')
  .option('--json', 'Output as JSON')
  .action(infoCommand);

// Settings commands
const settings = program
  .command('settings')
  .description('Manage framework and MCP server settings');

// Settings > Framework commands
const settingsFramework = settings
  .command('framework')
  .description('Manage framework settings');

settingsFramework
  .command('get <framework>')
  .description('Get settings for a framework')
  .option('--json', 'Output as JSON')
  .action(settingsFrameworkGet);

settingsFramework
  .command('set <framework>')
  .description('Update settings for a framework')
  .option('-t, --tools <list>', 'Comma-separated list of enabled tools')
  .option('--json', 'Output as JSON')
  .action(settingsFrameworkSet);

// Settings > MCP commands
const settingsMCP = settings
  .command('mcp')
  .description('Manage MCP servers');

settingsMCP
  .command('list')
  .description('List all MCP servers')
  .option('--json', 'Output as JSON')
  .action(settingsMCPList);

settingsMCP
  .command('add <name>')
  .description('Add a new MCP server')
  .requiredOption('-c, --command <cmd>', 'Command to run')
  .option('-a, --args <args>', 'Space-separated arguments')
  .option('-e, --env <json>', 'Environment variables as JSON')
  .option('--disabled', 'Add as disabled')
  .action(settingsMCPAdd);

settingsMCP
  .command('remove <name>')
  .description('Remove an MCP server')
  .action(settingsMCPRemove);

settingsMCP
  .command('enable <name>')
  .description('Enable an MCP server')
  .action((name) => settingsMCPToggle(name, true));

settingsMCP
  .command('disable <name>')
  .description('Disable an MCP server')
  .action((name) => settingsMCPToggle(name, false));

// Parse arguments
program.parse();
