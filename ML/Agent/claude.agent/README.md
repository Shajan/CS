# Claude Agent SDK Demo

A simple project demonstrating how to build conversational AI agents with Claude's API.

## Overview

This project shows you how to create AI agents that can:

- **Remember Context**: Maintain conversation history across multiple interactions
- **Have Natural Conversations**: Interact using natural language
- **Be Easily Extended**: Simple base classes you can customize for your needs
- **Work Out of the Box**: Ready-to-run demo with a clean UI

## What You'll Learn

- How to create a simple agent abstraction layer
- How to manage conversation history
- How to extend base agents for custom behavior
- How to integrate agents into a React application

## Prerequisites

- Node.js 18+
- pnpm (package manager)
- Anthropic API key

### Installing Node.js and pnpm

If you're new to Node.js, follow these steps:

#### 1. Install Node.js

**macOS:**
```bash
# Using Homebrew (recommended)
brew install node

# Or download from https://nodejs.org/
```

**Windows:**
- Download the installer from [nodejs.org](https://nodejs.org/)
- Run the installer and follow the prompts
- Node.js will include npm by default

**Linux (Ubuntu/Debian):**
```bash
# Using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

Verify installation:
```bash
node --version  # Should show v18 or higher
npm --version   # Should show npm version
```

#### 2. Install pnpm

Once Node.js is installed, install pnpm globally:

```bash
# Using npm (comes with Node.js)
npm install -g pnpm

# Or using the standalone installer
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

Verify pnpm installation:
```bash
pnpm --version
```

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
cd claude.agent
```

2. Install dependencies:
```bash
pnpm install
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

4. Start the development server (runs both frontend and backend):
```bash
pnpm dev
```

This will start:
- Backend server on `http://localhost:3001`
- Frontend dev server on `http://localhost:5173`

5. Open your browser and navigate to `http://localhost:5173`

### Why pnpm?

This project uses **pnpm** instead of npm because it:
- **Faster**: Installs packages significantly faster than npm
- **Efficient**: Saves disk space by storing packages in a single location
- **Strict**: Better dependency management and prevents phantom dependencies
- **Compatible**: Works with all npm packages and scripts

You can still use npm if you prefer, but pnpm is recommended for better performance.

## Project Structure

```
claude.agent/
├── server/                    # Backend (Express + Agent SDK)
│   ├── agents/                # Agent implementations
│   │   ├── base-agent.ts      # Base agent class (core abstraction)
│   │   ├── simple-chat-agent.ts # Example chat agent
│   │   └── index.ts           # Agent exports
│   ├── routes/
│   │   └── agent.ts           # API routes for agent interactions
│   └── index.ts               # Express server
├── src/                       # Frontend (React + TypeScript)
│   ├── App.tsx                # Main React component
│   ├── App.css                # Styles
│   ├── main.tsx               # React entry point
│   └── vite-env.d.ts          # TypeScript declarations
├── .env.example               # Environment variable template
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
└── vite.config.ts             # Vite configuration
```

## Agent Architecture

### Understanding the Agent SDK

The project includes a simple but powerful agent abstraction:

**`BaseAgent`** (`server/agents/base-agent.ts`):
- Core agent class that handles Claude API communication
- Manages conversation history automatically
- Provides simple `chat()` method for interaction
- Easy to extend for custom behavior

**`SimpleChatAgent`** (`server/agents/simple-chat-agent.ts`):
- Example of extending BaseAgent
- Adds system prompt functionality
- Shows how to build custom agents

### Building Your Own Agent

```typescript
import { BaseAgent, AgentConfig } from './agents';

// Extend BaseAgent for custom behavior
class MyCustomAgent extends BaseAgent {
  async chat(message: string): Promise<string> {
    // Add your custom logic here
    return super.chat(message);
  }
}

// Use your agent
const agent = new MyCustomAgent({ apiKey: process.env.ANTHROPIC_API_KEY! });
const response = await agent.chat('Hello!');
```

### API Endpoints

- **POST** `/api/agent/chat` - Send a message to the agent
- **POST** `/api/agent/clear` - Clear conversation history
- **GET** `/api/agent/history` - Get conversation history

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `PORT`: Server port (default: 3001)

## Technologies

**Frontend:**
- React 18
- TypeScript 5
- Vite

**Backend:**
- Node.js / Express
- Anthropic SDK (@anthropic-ai/sdk)
- TypeScript 5

## Learn More

- [Claude Agent SDK Documentation](https://github.com/anthropics/anthropic-sdk-typescript)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [React Documentation](https://react.dev/)

## License

MIT
