# Multi-Framework AI Agent System

A unified platform for testing and comparing multiple AI agent frameworks (Claude Agent SDK, LangChain, CrewAI, AutoGen, LlamaIndex, AutoGPT) through a common interface. Access via web UI or command-line interface.

## 🌟 Features

- **Framework Agnostic**: Use any supported agent framework through a unified API
- **Multiple Interfaces**: Access via web UI or CLI
- **Easy Framework Switching**: Switch between frameworks without code changes
- **Adapter Pattern**: Clean architecture for adding new frameworks
- **Session Management**: Maintain conversation history per framework
- **Framework Comparison**: Benchmark and compare frameworks side-by-side

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web UI Usage](#web-ui-usage)
- [CLI Usage](#cli-usage)
- [CLI Examples](#cli-examples)
- [Architecture](#architecture)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Adding New Frameworks](#adding-new-frameworks)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Node.js 18+
- pnpm (recommended) or npm

### Setup

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd frameworks
\`\`\`

2. Install dependencies:
\`\`\`bash
pnpm install
\`\`\`

3. Set up environment variables:
\`\`\`bash
cp .env.example .env
# Edit .env and add your API keys
\`\`\`

4. Build the project:
\`\`\`bash
pnpm build
\`\`\`

## ⚡ Quick Start

### Start the API Server

\`\`\`bash
cd common
pnpm install
pnpm dev
\`\`\`

The server will start on \`http://localhost:3001\`

### Start the Web UI

In a new terminal:

\`\`\`bash
cd ui
pnpm install
pnpm dev
\`\`\`

The UI will be available at \`http://localhost:5173\`

### Use the CLI

In a new terminal:

\`\`\`bash
cd cli
pnpm install

# Interactive chat
pnpm dev chat

# Single test
pnpm dev test -f claude-agent -m "What is 2+2?"

# List frameworks
pnpm dev list
\`\`\`

## 🖥️ Web UI Usage

### Accessing the UI

1. Start both the API server and UI (see Quick Start)
2. Open \`http://localhost:5173\` in your browser
3. Select a framework from the dropdown
4. Start chatting!

### Features

- **Framework Selector**: Choose from available frameworks
- **Chat Interface**: Send messages and receive responses
- **Message History**: View conversation history
- **Framework Info**: See capabilities and metadata
- **Clear History**: Reset conversation at any time

## 💻 CLI Usage

The CLI provides powerful commands for testing and comparing frameworks.

### Installation

\`\`\`bash
cd cli
pnpm install
pnpm build

# Optional: Link globally
npm link
\`\`\`

After linking, you can use \`agent-cli\` from anywhere.

### Commands

#### 1. Interactive Chat

Start an interactive chat session with a framework:

\`\`\`bash
agent-cli chat [options]

Options:
  -f, --framework <name>  Framework to use (default: "claude-agent")
  -s, --session <id>      Session ID (default: auto-generated)
\`\`\`

#### 2. Test Command

Test a framework with a single message:

\`\`\`bash
agent-cli test [options]

Options:
  -f, --framework <name>  Framework to use (required)
  -m, --message <text>    Message to send (required)
  -s, --session <id>      Session ID (optional)
  --json                  Output as JSON
\`\`\`

#### 3. Benchmark Command

Compare multiple frameworks with the same message:

\`\`\`bash
agent-cli benchmark [options]

Options:
  -m, --message <text>      Message to send (required)
  -f, --frameworks <list>   Comma-separated frameworks (default: all)
  --output <file>           Save results to file
  --json                    Output as JSON
\`\`\`

#### 4. List Command

List all available frameworks:

\`\`\`bash
agent-cli list [options]

Options:
  --json  Output as JSON
\`\`\`

#### 5. Info Command

Get details about a specific framework:

\`\`\`bash
agent-cli info <framework> [options]

Options:
  --json  Output as JSON
\`\`\`

## 📚 CLI Examples

### Example 1: Interactive Chat

\`\`\`bash
$ agent-cli chat -f claude-agent

🤖 Multi-Framework Agent Chat

Framework: claude-agent
Session: session-1705334567890

Commands: 'exit' to quit, 'clear' to clear history

You: What is the capital of France?
