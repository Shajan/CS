# Multi-Framework AI Agent System

A unified platform for testing and comparing multiple AI agent frameworks (Claude Agent SDK, LangChain, CrewAI, AutoGen, LlamaIndex, AutoGPT) through a common interface.

## Overview

This system provides a framework-agnostic way to interact with different AI agent implementations, allowing developers to:

- Test and compare multiple agent frameworks side-by-side
- Switch between frameworks without changing application code
- Access agents through Web UI, CLI, or REST API
- Benchmark performance across different implementations
- Maintain consistent interfaces while preserving framework-specific capabilities

### Supported Frameworks

| Framework | Status | Description |
|-----------|--------|-------------|
| **Claude Agent SDK** | ✅ Implemented | Anthropic's official agent SDK with session management |
| **LangChain** | 🚧 Planned | Vendor-agnostic framework with massive ecosystem |
| **CrewAI** | 🚧 Planned | Role-based multi-agent teams with SOP workflows |
| **AutoGen** | 🚧 Planned | Microsoft's event-driven multi-agent framework |
| **LlamaIndex** | 🚧 Planned | Document-focused agents with RAG capabilities |
| **AutoGPT** | 🚧 Planned | Autonomous agents with goal-pursuit behavior |

---

## Features

### Framework Agnostic
- Single API interface works with all frameworks
- Adapter pattern ensures consistent behavior
- Framework-specific capabilities preserved

### Multiple Interfaces
- **Web UI**: Interactive chat interface with framework selector
- **CLI**: Command-line tool for testing and automation
- **REST API**: Programmatic access for integration

### Adapter Pattern Architecture
- Clean separation between UI, API, and framework implementations
- Easy to add new frameworks by implementing standard interface
- Framework discovery and registration system

### Session Management
- Per-framework conversation history
- Session isolation and cleanup
- Resumable conversations

### Benchmarking
- Compare response quality across frameworks
- Measure performance metrics (latency, tokens)
- Export results for analysis

---

## Installation

### Prerequisites

- **Node.js**: 18.0.0 or higher
- **pnpm**: Package manager (install with `npm install -g pnpm`)
- **API Keys**: For the frameworks you want to use

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd frameworks
   ```

2. **Install dependencies**:
   ```bash
   pnpm install
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Build the project**:
   ```bash
   pnpm build
   ```

---

## Quick Start

### Start the API Server

```bash
cd common
pnpm dev
```

The API server will start on `http://localhost:3001`

### Start the Web UI

```bash
cd ui
pnpm install
pnpm dev
```

Open your browser to `http://localhost:5173`

### Use the CLI

```bash
cd cli
pnpm install
pnpm build

# Interactive chat
pnpm dev chat

# Or install globally
npm link
agent-cli chat
```

---

## Web UI Usage

### Accessing the Interface

1. Start the API server and Web UI (see Quick Start)
2. Navigate to `http://localhost:5173`
3. Select a framework from the dropdown menu
4. Start chatting!

### Framework Selection

- Click the framework dropdown in the header
- Choose from available frameworks
- Your selection is saved in browser storage
- Switching frameworks will clear the current conversation

### Chat Interface

- Type your message in the input box
- Press Enter or click Send
- View responses with framework metadata:
  - Model used
  - Tokens consumed
  - Response time

### Settings

- Configure API keys per framework
- Toggle frameworks on/off
- Set default framework
- Test framework connectivity

### Traces View

- View execution traces from agents
- Filter by framework
- Compare traces across frameworks
- Debug tool calls and reasoning

---

## CLI Usage

The CLI provides powerful commands for testing and comparing frameworks.

### Installation

```bash
cd cli
pnpm install
pnpm build
npm link  # Makes 'agent-cli' available globally
```

### Commands Overview

```bash
agent-cli --help

Commands:
  agent-cli chat        Start interactive chat session
  agent-cli test        Test framework with single message
  agent-cli benchmark   Compare frameworks with same prompt
  agent-cli list        List available frameworks
  agent-cli info        Get framework details

Options:
  --version             Show version number
  -h, --help           Show help
```

---

### `chat` - Interactive Chat

Start a conversational session with an agent framework.

#### Usage

```bash
agent-cli chat [options]
```

#### Options

```
-f, --framework <name>    Framework to use (default: "claude-agent")
-s, --session <id>        Session ID (default: random UUID)
-c, --config <json>       Framework config as JSON string
-h, --help               Display help
```

#### Examples

**Default framework (Claude Agent SDK)**:
```bash
agent-cli chat
```

**Specific framework**:
```bash
agent-cli chat --framework langchain
agent-cli chat -f crewai
```

**Custom session ID**:
```bash
agent-cli chat -f claude-agent -s my-session-123
```

**With framework configuration**:
```bash
agent-cli chat -f langchain -c '{"model":"gpt-4","temperature":0.7}'
```

#### Example Session

```
$ agent-cli chat -f claude-agent

> Using framework: Claude Agent SDK
> Session: abc-def-123
> Type 'exit' to quit, 'clear' to clear history, 'switch' to change framework

You: What is the capital of France?