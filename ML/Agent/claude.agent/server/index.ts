import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import agentRouter from './routes/agent.js';
import tracesRouter from './routes/traces.js';
import toolsRouter from './routes/tools.js';
import mcpRouter from './routes/mcp.js';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/agent', agentRouter);
app.use('/api/traces', tracesRouter);
app.use('/api/tools', toolsRouter);
app.use('/api/mcp', mcpRouter);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📡 API available at http://localhost:${PORT}/api/agent`);
});
