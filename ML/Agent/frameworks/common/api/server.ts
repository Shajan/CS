import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { adapterRegistry } from '../adapters/adapter-registry.js';
import chatRouter from './routes/chat.js';
import historyRouter from './routes/history.js';
import frameworksRouter from './routes/frameworks.js';
import tracesRouter from './routes/traces.js';
import settingsRouter from './routes/settings.js';

// Load environment variables from root directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
dotenv.config({ path: join(__dirname, '../../.env') });

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Request logging
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`);
  next();
});

// Routes
app.use('/api/chat', chatRouter);
app.use('/api/history', historyRouter);
app.use('/api/frameworks', frameworksRouter);
app.use('/api/traces', tracesRouter);
app.use('/api/settings', settingsRouter);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    adapters: adapterRegistry.count(),
  });
});

/**
 * Initialize and register adapters
 */
async function initializeAdapters() {
  console.log('\n🔧 Initializing adapters...\n');

  try {
    // Import and register Claude Agent adapter
    const { ClaudeAgentAdapter } = await import(
      '../../implementations/claude-agent/adapter.js'
    );
    const claudeAdapter = new ClaudeAgentAdapter();
    await claudeAdapter.initialize?.({});
    adapterRegistry.register(claudeAdapter);

    // Import and register LangChain adapter
    const { LangChainAdapter } = await import('../../implementations/langchain/adapter.js');
    const langchainAdapter = new LangChainAdapter();
    await langchainAdapter.initialize?.({});
    adapterRegistry.register(langchainAdapter);

    console.log(`\n✓ Registered ${adapterRegistry.count()} adapter(s)\n`);
  } catch (error) {
    console.error('Failed to initialize adapters:', error);
    throw error;
  }
}

/**
 * Start the server
 */
async function start() {
  try {
    await initializeAdapters();

    app.listen(PORT, () => {
      console.log(`\n${'='.repeat(50)}`);
      console.log(`🚀 Multi-Framework Agent API Server`);
      console.log(`${'='.repeat(50)}`);
      console.log(`\n📡 Server: http://localhost:${PORT}`);
      console.log(`🏥 Health:  http://localhost:${PORT}/health`);
      console.log(`🔧 API:     http://localhost:${PORT}/api`);
      console.log(`\n📚 Available endpoints:`);
      console.log(`   POST   /api/chat`);
      console.log(`   GET    /api/history?sessionId=xxx&framework=xxx`);
      console.log(`   DELETE /api/history?sessionId=xxx&framework=xxx`);
      console.log(`   GET    /api/frameworks`);
      console.log(`   GET    /api/frameworks/:name`);
      console.log(`   GET    /api/traces?sessionId=xxx&framework=xxx`);
      console.log(`   DELETE /api/traces?sessionId=xxx&framework=xxx`);
      console.log(`   GET    /api/traces/sessions`);
      console.log(`   GET    /api/settings`);
      console.log(`   GET    /api/settings/framework/:name`);
      console.log(`   PUT    /api/settings/framework/:name`);
      console.log(`   GET    /api/settings/mcp`);
      console.log(`   POST   /api/settings/mcp`);
      console.log(`   PUT    /api/settings/mcp/:name`);
      console.log(`   DELETE /api/settings/mcp/:name`);
      console.log(`\n${'='.repeat(50)}\n`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n\n🛑 Shutting down server...');
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n\n🛑 Shutting down server...');
  process.exit(0);
});

// Start the server
start();
