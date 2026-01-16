import { MockAdapter } from '../mocks/mock-adapter.js';
import type { Message } from '../../types/index.js';

describe('BaseAdapter', () => {
  let adapter: MockAdapter;

  beforeEach(() => {
    adapter = new MockAdapter();
  });

  describe('session management', () => {
    it('should create new session on first access', async () => {
      const sessionId = 'test-session-1';
      const history = await adapter.getHistory(sessionId);

      expect(history.sessionId).toBe(sessionId);
      expect(history.messages).toEqual([]);
      expect(history.framework).toBe('mock');
      expect(history.startedAt).toBeInstanceOf(Date);
      expect(history.lastActiveAt).toBeInstanceOf(Date);
    });

    it('should return same session on subsequent accesses', async () => {
      const sessionId = 'test-session-2';

      const history1 = await adapter.getHistory(sessionId);
      const history2 = await adapter.getHistory(sessionId);

      expect(history1).toBe(history2);
    });

    it('should maintain separate sessions for different IDs', async () => {
      const session1 = await adapter.getHistory('session-1');
      const session2 = await adapter.getHistory('session-2');

      expect(session1.sessionId).toBe('session-1');
      expect(session2.sessionId).toBe('session-2');
      expect(session1).not.toBe(session2);
    });
  });

  describe('getHistory', () => {
    it('should return empty history for new session', async () => {
      const history = await adapter.getHistory('new-session');

      expect(history.messages).toEqual([]);
    });

    it('should return messages after chat', async () => {
      const sessionId = 'chat-session';

      await adapter.chat({
        message: 'Hello',
        sessionId,
      });

      const history = await adapter.getHistory(sessionId);

      expect(history.messages).toHaveLength(2);
      expect(history.messages[0].role).toBe('user');
      expect(history.messages[0].content).toBe('Hello');
      expect(history.messages[1].role).toBe('assistant');
      expect(history.messages[1].content).toBe('Mock response');
    });

    it('should accumulate messages across multiple chats', async () => {
      const sessionId = 'multi-chat';

      await adapter.chat({ message: 'First', sessionId });
      await adapter.chat({ message: 'Second', sessionId });
      await adapter.chat({ message: 'Third', sessionId });

      const history = await adapter.getHistory(sessionId);

      expect(history.messages).toHaveLength(6); // 3 user + 3 assistant
      expect(history.messages[0].content).toBe('First');
      expect(history.messages[2].content).toBe('Second');
      expect(history.messages[4].content).toBe('Third');
    });

    it('should update lastActiveAt on message addition', async () => {
      const sessionId = 'active-session';

      const history1 = await adapter.getHistory(sessionId);
      const firstTime = history1.lastActiveAt;

      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 10));

      await adapter.chat({ message: 'Hello', sessionId });

      const history2 = await adapter.getHistory(sessionId);
      const secondTime = history2.lastActiveAt;

      expect(secondTime!.getTime()).toBeGreaterThan(firstTime!.getTime());
    });
  });

  describe('clearHistory', () => {
    it('should clear session history', async () => {
      const sessionId = 'clear-test';

      // Add some messages
      await adapter.chat({ message: 'Message 1', sessionId });
      await adapter.chat({ message: 'Message 2', sessionId });

      let history = await adapter.getHistory(sessionId);
      expect(history.messages).toHaveLength(4);

      // Clear history
      await adapter.clearHistory(sessionId);

      // Get history again - should be empty
      history = await adapter.getHistory(sessionId);
      expect(history.messages).toEqual([]);
    });

    it('should not affect other sessions', async () => {
      const session1 = 'session-1';
      const session2 = 'session-2';

      await adapter.chat({ message: 'Message 1', sessionId: session1 });
      await adapter.chat({ message: 'Message 2', sessionId: session2 });

      await adapter.clearHistory(session1);

      const history1 = await adapter.getHistory(session1);
      const history2 = await adapter.getHistory(session2);

      expect(history1.messages).toEqual([]);
      expect(history2.messages).toHaveLength(2);
    });

    it('should not throw error when clearing non-existent session', async () => {
      await expect(
        adapter.clearHistory('non-existent')
      ).resolves.not.toThrow();
    });
  });

  describe('chat integration', () => {
    it('should track all chat calls', async () => {
      adapter.resetCalls();

      await adapter.chat({ message: 'Test 1', sessionId: 's1' });
      await adapter.chat({ message: 'Test 2', sessionId: 's2' });

      expect(adapter.chatCalls).toHaveLength(2);
      expect(adapter.chatCalls[0].message).toBe('Test 1');
      expect(adapter.chatCalls[1].message).toBe('Test 2');
    });

    it('should track history operations', async () => {
      adapter.resetCalls();

      await adapter.getHistory('session-1');
      await adapter.clearHistory('session-1');

      expect(adapter.getHistoryCalls).toContain('session-1');
      expect(adapter.clearHistoryCalls).toContain('session-1');
    });
  });

  describe('logging', () => {
    it('should log messages without throwing errors', () => {
      expect(() => {
        adapter['log']('Test message', 'arg1', 'arg2');
      }).not.toThrow();
    });

    it('should log errors without throwing errors', () => {
      const error = new Error('Test error');
      expect(() => {
        adapter['logError']('Error occurred', error);
      }).not.toThrow();
    });
  });
});
