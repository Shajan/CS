import { describe, it, expect, beforeEach } from '@jest/globals';
import { ClaudeAgentAdapter } from '../adapter.js';

describe('ClaudeAgentAdapter', () => {
  let adapter: ClaudeAgentAdapter;
  const testApiKey = 'test-api-key-12345';

  beforeEach(() => {
    adapter = new ClaudeAgentAdapter(testApiKey);
  });

  describe('metadata', () => {
    it('should have correct adapter metadata', () => {
      expect(adapter.name).toBe('claude-agent');
      expect(adapter.displayName).toBe('Claude Agent SDK');
      expect(adapter.version).toBe('1.0.0');
      expect(adapter.description).toBeTruthy();
    });
  });

  describe('initialize', () => {
    it('should initialize without errors', async () => {
      await expect(adapter.initialize({})).resolves.not.toThrow();
    });

    it('should accept custom API key during initialization', async () => {
      const customAdapter = new ClaudeAgentAdapter(testApiKey);
      await expect(
        customAdapter.initialize({ apiKey: 'custom-key' })
      ).resolves.not.toThrow();
    });
  });

  describe('history management', () => {
    it('should return empty history for non-existent session', async () => {
      const history = await adapter.getHistory('non-existent');

      expect(history.sessionId).toBe('non-existent');
      expect(history.messages).toHaveLength(0);
      expect(history.framework).toBe('claude-agent');
    });

    it('should clear conversation history for non-existent session', async () => {
      await expect(adapter.clearHistory('test-session')).resolves.not.toThrow();
    });
  });

  describe('lifecycle', () => {
    it('should shutdown gracefully', async () => {
      await expect(adapter.shutdown()).resolves.not.toThrow();
    });

    it('should pass health check with API key', async () => {
      const healthy = await adapter.healthCheck();
      expect(healthy).toBe(true);
    });
  });

  describe('capabilities', () => {
    it('should return framework capabilities', () => {
      const capabilities = adapter.getCapabilities();

      expect(capabilities).toBeDefined();
      expect(capabilities.supportsTools).toBe(true);
      expect(capabilities.supportsMultiModal).toBe(true);
      expect(capabilities.maxContextLength).toBe(200000);
      expect(capabilities.supportedModels).toContain('claude-sonnet-4-5-20250929');
    });

    it('should return current configuration', () => {
      const config = adapter.getConfiguration();

      expect(config).toBeDefined();
      expect(config.model).toBe('claude-sonnet-4-5-20250929');
      expect(config.maxTokens).toBe(1024);
      expect(config.hasApiKey).toBe(true);
    });
  });

  describe('error handling', () => {
    it('should throw error if no API key provided', () => {
      expect(() => new ClaudeAgentAdapter('')).toThrow('ANTHROPIC_API_KEY is required');
    });

    it('should create adapter with valid API key', () => {
      expect(() => new ClaudeAgentAdapter('valid-key')).not.toThrow();
    });
  });

  describe('interface compliance', () => {
    it('should implement all required AgentAdapter methods', () => {
      expect(typeof adapter.chat).toBe('function');
      expect(typeof adapter.clearHistory).toBe('function');
      expect(typeof adapter.getHistory).toBe('function');
    });

    it('should implement all optional AgentAdapter methods', () => {
      expect(typeof adapter.initialize).toBe('function');
      expect(typeof adapter.shutdown).toBe('function');
      expect(typeof adapter.healthCheck).toBe('function');
      expect(typeof adapter.getCapabilities).toBe('function');
      expect(typeof adapter.getConfiguration).toBe('function');
    });
  });
});
