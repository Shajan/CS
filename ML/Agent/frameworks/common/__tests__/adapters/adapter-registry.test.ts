import { AdapterRegistry } from '../../adapters/adapter-registry.js';
import { MockAdapter } from '../mocks/mock-adapter.js';
import { BaseAdapter } from '../../adapters/base-adapter.js';
import type { AgentAdapter } from '../../adapters/index.js';
import type { AgentRequest, AgentResponse, ConversationHistory } from '../../types/index.js';

// Helper to create a second mock adapter with different name
class SecondMockAdapter extends BaseAdapter implements AgentAdapter {
  readonly name = 'mock2';
  readonly displayName = 'Mock Adapter 2';
  readonly version = '1.0.0';

  async chat(request: AgentRequest): Promise<AgentResponse> {
    return {
      response: 'Mock 2 response',
      sessionId: request.sessionId,
      framework: this.name,
    };
  }
}

describe('AdapterRegistry', () => {
  let registry: AdapterRegistry;
  let mockAdapter: MockAdapter;

  beforeEach(() => {
    registry = new AdapterRegistry();
    mockAdapter = new MockAdapter();
  });

  describe('register', () => {
    it('should register an adapter successfully', () => {
      expect(() => registry.register(mockAdapter)).not.toThrow();
      expect(registry.count()).toBe(1);
    });

    it('should throw error when registering duplicate adapter', () => {
      registry.register(mockAdapter);
      const duplicateAdapter = new MockAdapter();

      expect(() => registry.register(duplicateAdapter)).toThrow(
        "Adapter 'mock' is already registered"
      );
    });

    it('should allow registering multiple different adapters', () => {
      const adapter1 = new MockAdapter();
      const adapter2 = new SecondMockAdapter();

      registry.register(adapter1);
      registry.register(adapter2);

      expect(registry.count()).toBe(2);
    });
  });

  describe('getAdapter', () => {
    it('should retrieve registered adapter by name', () => {
      registry.register(mockAdapter);
      const retrieved = registry.getAdapter('mock');

      expect(retrieved).toBe(mockAdapter);
      expect(retrieved.name).toBe('mock');
    });

    it('should throw error when adapter not found', () => {
      expect(() => registry.getAdapter('nonexistent')).toThrow(
        "Adapter 'nonexistent' not found"
      );
    });

    it('should include available adapters in error message', () => {
      registry.register(mockAdapter);

      expect(() => registry.getAdapter('nonexistent')).toThrow(
        'Available adapters: mock'
      );
    });
  });

  describe('hasAdapter', () => {
    it('should return true for registered adapter', () => {
      registry.register(mockAdapter);
      expect(registry.hasAdapter('mock')).toBe(true);
    });

    it('should return false for unregistered adapter', () => {
      expect(registry.hasAdapter('nonexistent')).toBe(false);
    });
  });

  describe('listAdapters', () => {
    it('should return empty array when no adapters registered', () => {
      const adapters = registry.listAdapters();
      expect(adapters).toEqual([]);
    });

    it('should return adapter information for registered adapters', () => {
      registry.register(mockAdapter);
      const adapters = registry.listAdapters();

      expect(adapters).toHaveLength(1);
      expect(adapters[0]).toEqual({
        name: 'mock',
        displayName: 'Mock Adapter',
        version: '1.0.0',
        description: 'A mock adapter for testing',
        capabilities: mockAdapter.getCapabilities(),
      });
    });

    it('should return multiple adapter information', () => {
      const adapter1 = new MockAdapter();
      const adapter2 = new SecondMockAdapter();

      registry.register(adapter1);
      registry.register(adapter2);

      const adapters = registry.listAdapters();
      expect(adapters).toHaveLength(2);
      expect(adapters.map(a => a.name)).toContain('mock');
      expect(adapters.map(a => a.name)).toContain('mock2');
    });
  });

  describe('count', () => {
    it('should return 0 for empty registry', () => {
      expect(registry.count()).toBe(0);
    });

    it('should return correct count after registrations', () => {
      expect(registry.count()).toBe(0);

      registry.register(mockAdapter);
      expect(registry.count()).toBe(1);

      const adapter2 = new SecondMockAdapter();
      registry.register(adapter2);
      expect(registry.count()).toBe(2);
    });
  });

  describe('unregister', () => {
    it('should remove registered adapter', () => {
      registry.register(mockAdapter);
      expect(registry.count()).toBe(1);

      const result = registry.unregister('mock');
      expect(result).toBe(true);
      expect(registry.count()).toBe(0);
    });

    it('should return false when unregistering non-existent adapter', () => {
      const result = registry.unregister('nonexistent');
      expect(result).toBe(false);
    });

    it('should allow re-registration after unregister', () => {
      registry.register(mockAdapter);
      registry.unregister('mock');

      expect(() => registry.register(mockAdapter)).not.toThrow();
      expect(registry.count()).toBe(1);
    });
  });

  describe('clear', () => {
    it('should remove all adapters', () => {
      const adapter1 = new MockAdapter();
      const adapter2 = new SecondMockAdapter();

      registry.register(adapter1);
      registry.register(adapter2);
      expect(registry.count()).toBe(2);

      registry.clear();
      expect(registry.count()).toBe(0);
    });

    it('should work on empty registry', () => {
      expect(() => registry.clear()).not.toThrow();
      expect(registry.count()).toBe(0);
    });
  });

  describe('integration', () => {
    it('should handle complete registration lifecycle', () => {
      // Register
      registry.register(mockAdapter);
      expect(registry.hasAdapter('mock')).toBe(true);

      // Retrieve
      const retrieved = registry.getAdapter('mock');
      expect(retrieved).toBe(mockAdapter);

      // List
      const adapters = registry.listAdapters();
      expect(adapters).toHaveLength(1);

      // Unregister
      registry.unregister('mock');
      expect(registry.hasAdapter('mock')).toBe(false);
    });
  });
});
