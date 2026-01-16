import chalk from 'chalk';

/**
 * Print success message
 */
export function success(message: string): void {
  console.log(chalk.green('✓'), message);
}

/**
 * Print error message
 */
export function error(message: string): void {
  console.error(chalk.red('✗'), message);
}

/**
 * Print info message
 */
export function info(message: string): void {
  console.log(chalk.blue('ℹ'), message);
}

/**
 * Print warning message
 */
export function warning(message: string): void {
  console.log(chalk.yellow('⚠'), message);
}

/**
 * Print a header
 */
export function header(message: string): void {
  console.log();
  console.log(chalk.bold.cyan(message));
  console.log(chalk.cyan('='.repeat(message.length)));
  console.log();
}

/**
 * Format a table row
 */
export function tableRow(columns: string[], widths: number[]): string {
  return columns
    .map((col, i) => col.padEnd(widths[i]))
    .join(' │ ');
}

/**
 * Print a table
 */
export function table(headers: string[], rows: string[][], widths?: number[]): void {
  // Calculate column widths if not provided
  const colWidths =
    widths ||
    headers.map((header, i) => {
      const maxRowWidth = Math.max(...rows.map((row) => row[i]?.length || 0));
      return Math.max(header.length, maxRowWidth) + 2;
    });

  // Print header
  console.log(chalk.bold(tableRow(headers, colWidths)));
  console.log('─'.repeat(colWidths.reduce((a, b) => a + b + 3, 0)));

  // Print rows
  rows.forEach((row) => {
    console.log(tableRow(row, colWidths));
  });
  console.log();
}

/**
 * Format metadata
 */
export function formatMetadata(metadata?: Record<string, any>): string {
  if (!metadata) return '';

  const parts: string[] = [];

  if (metadata.model) {
    parts.push(chalk.dim(`Model: ${metadata.model}`));
  }

  if (metadata.tokensUsed) {
    parts.push(chalk.dim(`Tokens: ${metadata.tokensUsed}`));
  }

  if (metadata.duration) {
    parts.push(chalk.dim(`Duration: ${metadata.duration}ms`));
  }

  return parts.length > 0 ? `[${parts.join(' | ')}]` : '';
}
