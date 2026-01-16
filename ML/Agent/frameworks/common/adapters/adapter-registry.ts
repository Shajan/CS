import type { AgentAdapter } from './adapter.interface.js';
import type { AdapterInfo } from '../types/index.js';

/**
 * AdapterRegistry
 *
 * Central registry for managing framework adapters. Provides:
 * - Registration of new adapters
 * - Retrieval of adapters by name
 * - Listing of all available adapters
 *
 * This class implements the Factory pattern for framework routing.
 */
export class AdapterRegistry {
  private adapters: Map<string, AgentAdapter> = new Map();

  /**
   * Register a framework adapter
   *
   * @param adapter - The adapter instance to register
   * @throws Error if adapter with same name already registered
   */
  register(adapter: AgentAdapter): void {
    if (this.adapters.has(adapter.name)) {
      throw new Error(`Adapter '${adapter.name}' is already registered`);
    }

    this.adapters.set(adapter.name, adapter);
    console.log(`✓ Registered adapter: ${adapter.displayName} (${adapter.name})`);
  }

  /**
   * Get an adapter by name
   *
   * @param name - Machine-readable adapter name
   * @returns The adapter instance
   * @throws Error if adapter not found
   */
  getAdapter(name: string): AgentAdapter {
    const adapter = this.adapters.get(name);
    if (!adapter) {
      const available = Array.from(this.adapters.keys()).join(', ');
      throw new Error(
        `Adapter '${name}' not found. Available adapters: ${available || 'none'}`
      );
    }
    return adapter;
  }

  /**
   * Check if an adapter is registered
   *
   * @param name - Machine-readable adapter name
   * @returns true if adapter exists, false otherwise
   */
  hasAdapter(name: string): boolean {
    return this.adapters.has(name);
  }

  /**
   * List all registered adapters
   *
   * @returns Array of adapter information
   */
  listAdapters(): AdapterInfo[] {
    return Array.from(this.adapters.values()).map((adapter) => ({
      name: adapter.name,
      displayName: adapter.displayName,
      version: adapter.version,
      description: adapter.description,
      capabilities: adapter.getCapabilities?.(),
    }));
  }

  /**
   * Get count of registered adapters
   *
   * @returns Number of registered adapters
   */
  count(): number {
    return this.adapters.size;
  }

  /**
   * Unregister an adapter (useful for testing)
   *
   * @param name - Machine-readable adapter name
   * @returns true if adapter was removed, false if not found
   */
  unregister(name: string): boolean {
    return this.adapters.delete(name);
  }

  /**
   * Clear all registered adapters
   */
  clear(): void {
    this.adapters.clear();
  }
}

// Export a singleton instance
export const adapterRegistry = new AdapterRegistry();
