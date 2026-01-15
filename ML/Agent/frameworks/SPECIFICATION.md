# Multi-Framework AI Agent System - Technical Specification

**Version**: 1.0
**Date**: January 15, 2026
**Status**: Planning

## 1. Overview

### 1.1 Purpose
Build a unified system that allows testing and comparing multiple AI agent frameworks (Claude Agent SDK, LangChain, CrewAI, AutoGen, LlamaIndex, AutoGPT) through a common interface, accessible via both web UI and command-line interface.

### 1.2 Goals
- Separate UI, API, and framework-specific implementations
- Provide standardized API layer that works with any framework
- Enable framework selection from UI and CLI
- Maintain framework-specific capabilities without constraint
- Support easy addition of new frameworks

### 1.3 Non-Goals
- Deep integration between multiple frameworks simultaneously
- Production-grade deployment configuration
- Framework-specific optimization beyond their native capabilities

## 2. Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Clients                                   │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   React Web UI       │      │   CLI Application     │    │
│  │  - Framework Select  │      │  - Interactive Chat   │    │
│  │  - Chat Interface    │      │  - Testing Commands   │    │
│  │  - Settings/Traces   │      │  - Benchmarking       │    │
│  └──────────────────────┘      └──────────────────────┘    │
└────────────────┬──────────────────────────┬─────────────────┘
                 │                          │
                 │      HTTP/REST API       │
                 │                          │
┌────────────────┴──────────────────────────┴─────────────────┐
│              Common API Layer (Express)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Routes: /api/chat, /api/history, /api/frameworks   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Adapter Registry (Factory Pattern)            │  │
│  │  - Discovers and registers framework adapters        │  │
│  │  - Routes requests to correct adapter                │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────┬──────────────────────────┬─────────────────┘
                 │                          │
        ┌────────┴────────┐        ┌───────┴────────┐
        │ AgentAdapter    │        │ AgentAdapter   │
        │   Interface     │  ...   │   Interface    │
        └────────┬────────┘        └───────┬────────┘
                 │                          │
┌────────────────┴──────────────────────────┴─────────────────┐
│                Framework Implementations                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Claude   │ │LangChain │ │ CrewAI   │ │ AutoGen  │ ...  │
│  │  Agent   │ │          │ │          │ │          │      │
│  │   SDK    │ │          │ │          │ │          │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
frameworks/
├── common/                          # Shared code
│   ├── types/                       # TypeScript interfaces
│   │   ├── agent.types.ts          # Core types
│   │   ├── message.types.ts        # Message structures
│   │   └── config.types.ts         # Configuration types
│   ├── api/                         # Framework-agnostic API
│   │   ├── server.ts               # Express server setup
│   │   ├── routes/
│   │   │   ├── chat.ts            # POST /api/chat
│   │   │   ├── history.ts         # GET/DELETE /api/history
│   │   │   └── frameworks.ts      # GET /api/frameworks
│   │   └── middleware/
│   │       └── framework-selector.ts
│   └── adapters/                   # Adapter infrastructure
│       ├── adapter.interface.ts   # AgentAdapter interface
│       ├── adapter-registry.ts    # Registry/factory
│       └── base-adapter.ts        # Optional base class
│
├── implementations/                # Framework-specific code
│   ├── claude-agent/              # Anthropic Claude Agent SDK
│   │   ├── adapter.ts
│   │   ├── agents/
│   │   ├── tools/
│   │   └── config.json
│   ├── langchain/                 # LangChain/LangGraph
│   │   ├── adapter.ts
│   │   ├── chains/
│   │   └── config.json
│   ├── crewai/                    # CrewAI
│   │   ├── adapter.ts
│   │   ├── crews/
│   │   └── config.json
│   ├── autogen/                   # Microsoft AutoGen
│   │   ├── adapter.ts
│   │   └── config.json
│   ├── llamaindex/                # LlamaIndex
│   │   ├── adapter.ts
│   │   └── config.json
│   └── autogpt/                   # AutoGPT
│       ├── adapter.ts
│       └── config.json
│
├── ui/                            # React web interface
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx
│   │   │   ├── FrameworkSelector.tsx
│   │   │   ├── Settings.tsx
│   │   │   └── Traces.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── cli/                           # Command-line interface
│   ├── src/
│   │   ├── index.ts              # Main CLI entry
│   │   ├── commands/
│   │   │   ├── chat.ts          # Interactive mode
│   │   │   ├── test.ts          # Single message test
│   │   │   └── benchmark.ts     # Compare frameworks
│   │   └── utils/
│   │       └── output.ts        # Formatting helpers
│   └── package.json
│
├── SPECIFICATION.md               # This document
├── STATUS.md                      # Progress tracking
├── .env.example                   # Environment template
└── package.json                   # Root workspace config
```

## 3. Core Abstractions

### 3.1 AgentAdapter Interface

Every framework must implement this interface:

```typescript
export interface AgentAdapter {
  // Metadata
  readonly name: string;              // Machine name: "claude-agent"
  readonly displayName: string;       // Human name: "Claude Agent SDK"
  readonly version: string;           // Adapter version
  readonly description?: string;      // Brief description

  // Core methods (REQUIRED)
  chat(request: AgentRequest): Promise<AgentResponse>;
  clearHistory(sessionId: string): Promise<void>;
  getHistory(sessionId: string): Promise<ConversationHistory>;

  // Lifecycle methods (OPTIONAL)
  initialize?(config: Record<string, any>): Promise<void>;
  shutdown?(): Promise<void>;
  healthCheck?(): Promise<boolean>;

  // Metadata methods (OPTIONAL)
  getCapabilities?(): AdapterCapabilities;
  getConfiguration?(): Record<string, any>;
}
```

### 3.2 Core Types

```typescript
// Request sent to any framework
export interface AgentRequest {
  message: string;                    // User message
  sessionId: string;                  // Session identifier
  framework?: string;                 // Framework name (for routing)
  config?: Record<string, any>;       // Framework-specific config
  metadata?: Record<string, any>;     // Request metadata
}

// Response from any framework
export interface AgentResponse {
  response: string;                   // Assistant's response
  sessionId: string;                  // Session identifier
  framework: string;                  // Which framework processed this
  metadata?: {
    model?: string;                   // Model used
    tokensUsed?: number;              // Token count
    duration?: number;                // Response time (ms)
    [key: string]: any;               // Framework-specific data
  };
}

// Message structure
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
  metadata?: Record<string, any>;
}

// Conversation history
export interface ConversationHistory {
  sessionId: string;
  messages: Message[];
  framework?: string;
  startedAt?: Date;
  lastActiveAt?: Date;
}

// Framework capabilities
export interface AdapterCapabilities {
  supportsStreaming?: boolean;        // SSE/streaming responses
  supportsTools?: boolean;            // Tool/function calling
  supportsMultiModal?: boolean;       // Images, files, etc.
  supportsMultiAgent?: boolean;       // Multiple cooperating agents
  supportsMemory?: boolean;           // Long-term memory
  maxContextLength?: number;          // Token limit
  supportedModels?: string[];         // Available models
}
```

### 3.3 Adapter Registry

```typescript
class AdapterRegistry {
  // Register a framework adapter
  register(adapter: AgentAdapter): void;

  // Get adapter by name
  getAdapter(name: string): AgentAdapter;

  // List all registered adapters
  listAdapters(): AdapterInfo[];

  // Check if adapter exists
  hasAdapter(name: string): boolean;

  // Auto-discover and register adapters from implementations/
  autoDiscoverAdapters(): Promise<void>;
}

interface AdapterInfo {
  name: string;
  displayName: string;
  version: string;
  description?: string;
  capabilities?: AdapterCapabilities;
}
```

## 4. API Specifications

### 4.1 REST Endpoints

#### POST /api/chat
Send a message to an agent.

**Request:**
```json
{
  "message": "What is the weather?",
  "sessionId": "user-123",
  "framework": "claude-agent",
  "config": {}
}
```

**Response:**
```json
{
  "response": "I don't have access to real-time weather...",
  "sessionId": "user-123",
  "framework": "claude-agent",
  "metadata": {
    "model": "claude-sonnet-4-5-20250929",
    "tokensUsed": 150,
    "duration": 1250
  }
}
```

#### GET /api/history?sessionId=user-123
Get conversation history for a session.

**Response:**
```json
{
  "sessionId": "user-123",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2026-01-15T10:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help?",
      "timestamp": "2026-01-15T10:00:01Z"
    }
  ],
  "framework": "claude-agent"
}
```

#### DELETE /api/history?sessionId=user-123
Clear conversation history.

**Response:**
```json
{
  "success": true,
  "sessionId": "user-123"
}
```

#### GET /api/frameworks
List all available frameworks.

**Response:**
```json
{
  "frameworks": [
    {
      "name": "claude-agent",
      "displayName": "Claude Agent SDK",
      "version": "1.0.0",
      "description": "Anthropic's official agent SDK",
      "capabilities": {
        "supportsStreaming": true,
        "supportsTools": true,
        "maxContextLength": 200000
      }
    },
    {
      "name": "langchain",
      "displayName": "LangChain",
      "version": "1.0.0"
    }
  ]
}
```

#### GET /api/frameworks/:name
Get details about a specific framework.

**Response:**
```json
{
  "name": "claude-agent",
  "displayName": "Claude Agent SDK",
  "version": "1.0.0",
  "description": "Anthropic's official agent SDK",
  "capabilities": {
    "supportsStreaming": true,
    "supportsTools": true,
    "supportsMultiModal": true,
    "maxContextLength": 200000,
    "supportedModels": ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101"]
  }
}
```

#### GET /health
API health check.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-01-15T10:00:00Z"
}
```

## 5. CLI Specifications

### 5.1 Commands

#### agent-cli chat
Start an interactive chat session.

```bash
# Use default framework (claude-agent)
agent-cli chat

# Use specific framework
agent-cli chat --framework langchain

# With custom session ID
agent-cli chat -f crewai -s my-session

Options:
  -f, --framework <name>    Framework to use (default: "claude-agent")
  -s, --session <id>        Session ID (default: random)
  -c, --config <json>       Framework config as JSON string
```

#### agent-cli test
Test a framework with a single message.

```bash
agent-cli test -f claude-agent -m "What is 2+2?"

Options:
  -f, --framework <name>    Framework to use (required)
  -m, --message <text>      Message to send (required)
  -s, --session <id>        Session ID (default: random)
  --json                    Output as JSON
```

#### agent-cli benchmark
Compare all frameworks with the same prompt.

```bash
agent-cli benchmark -m "Explain quantum computing"

Options:
  -m, --message <text>      Message to send (required)
  -f, --frameworks <list>   Comma-separated frameworks (default: all)
  --output <file>           Save results to file
  --json                    Output as JSON
```

#### agent-cli list
List all available frameworks.

```bash
agent-cli list

Options:
  --json                    Output as JSON
```

#### agent-cli info
Get details about a framework.

```bash
agent-cli info claude-agent

Options:
  --json                    Output as JSON
```

### 5.2 CLI Output Format

**Interactive Chat:**
```
> Using framework: Claude Agent SDK
> Session: abc-123
> Type 'exit' to quit, 'clear' to clear history

You: What is 2+2?
Assistant: The sum of 2+2 equals 4.
[claude-agent | 150 tokens | 1.2s]

You: exit
```

**Test Command:**
```
Framework: claude-agent
Message: What is 2+2?
Response: The sum of 2+2 equals 4.
Metadata:
  Model: claude-sonnet-4-5-20250929
  Tokens: 150
  Duration: 1250ms
```

**Benchmark Command:**
```
Testing 3 frameworks with message: "Explain quantum computing"

claude-agent:    ✓ 2.5s | 450 tokens
langchain:       ✓ 3.1s | 520 tokens
crewai:          ✗ Error: Not configured

Results saved to benchmark-2026-01-15.json
```

## 6. UI Specifications

### 6.1 Components

#### FrameworkSelector Component
- Dropdown showing all available frameworks
- Display framework name and description
- Show capabilities badges (tools, streaming, etc.)
- Remember last selection in localStorage

#### Chat Component (Enhanced)
- Add framework selector at top
- Show current framework in chat header
- Display framework-specific metadata in message footer
- Support framework switching mid-conversation (warn about history loss)

#### Settings Component (Enhanced)
- Per-framework configuration sections
- Framework-specific API keys and settings
- Toggle frameworks on/off
- Test framework connectivity

#### Traces Component (Enhanced)
- Filter traces by framework
- Show framework-specific trace data
- Compare traces across frameworks

### 6.2 State Management

```typescript
interface AppState {
  selectedFramework: string;
  availableFrameworks: AdapterInfo[];
  currentSession: string;
  conversation: Message[];
  frameworkConfigs: Record<string, any>;
  settings: {
    defaultFramework: string;
    enabledFrameworks: string[];
  };
}
```

## 7. Implementation Requirements

### 7.1 Phase 1: Foundation
- [ ] Create common types and interfaces
- [ ] Implement AgentAdapter interface
- [ ] Build AdapterRegistry
- [ ] Set up project structure

### 7.2 Phase 2: Refactor Claude Agent
- [ ] Move existing claude.agent code to implementations/claude-agent/
- [ ] Create ClaudeAgentAdapter
- [ ] Test adapter with existing functionality
- [ ] Update tests

### 7.3 Phase 3: Common API Layer
- [ ] Build Express server with routes
- [ ] Implement framework routing
- [ ] Add middleware
- [ ] Create API documentation

### 7.4 Phase 4: UI Updates
- [ ] Create FrameworkSelector component
- [ ] Update Chat component with framework selection
- [ ] Update Settings for multi-framework config
- [ ] Add framework switching logic

### 7.5 Phase 5: CLI Development
- [ ] Set up CLI with commander
- [ ] Implement chat command
- [ ] Implement test command
- [ ] Implement benchmark command
- [ ] Add output formatting

### 7.6 Phase 6: Additional Frameworks
- [ ] Add LangChain adapter
- [ ] Add CrewAI adapter
- [ ] Add AutoGen adapter
- [ ] Add LlamaIndex adapter
- [ ] Add AutoGPT adapter

## 8. Configuration

### 8.1 Environment Variables

```bash
# Common
NODE_ENV=development
PORT=3001
LOG_LEVEL=info

# Claude Agent SDK
ANTHROPIC_API_KEY=sk-ant-xxx

# LangChain
OPENAI_API_KEY=sk-xxx
LANGCHAIN_API_KEY=xxx

# Other frameworks
CREWAI_API_KEY=xxx
AUTOGEN_API_KEY=xxx
# ... etc
```

### 8.2 Framework Config Files

Each framework has a `config.json`:

```json
{
  "name": "claude-agent",
  "displayName": "Claude Agent SDK",
  "version": "1.0.0",
  "description": "Anthropic's official agent SDK",
  "enabled": true,
  "requiresApiKey": true,
  "envVars": ["ANTHROPIC_API_KEY"],
  "defaultConfig": {
    "model": "claude-sonnet-4-5-20250929",
    "maxTokens": 1024,
    "temperature": 1.0
  }
}
```

## 9. Testing Strategy

### 9.1 Unit Tests
- Test each adapter independently
- Test adapter registry
- Test API routes
- Test CLI commands

### 9.2 Integration Tests
- Test UI with API
- Test CLI with API
- Test framework switching
- Test error handling

### 9.3 Framework Tests
- Standard test suite that all adapters must pass
- Test basic chat functionality
- Test history management
- Test error scenarios

## 10. Success Criteria

- ✅ UI can select and switch between frameworks
- ✅ CLI can test any framework with same commands
- ✅ API layer is completely framework-agnostic
- ✅ Adding a new framework requires only implementing adapter interface
- ✅ Each framework maintains its native capabilities
- ✅ All endpoints documented and working
- ✅ At least 3 frameworks fully implemented

## 11. Future Enhancements

- Streaming support for real-time responses
- Multi-agent conversations (agents talking to each other)
- Framework comparison dashboard
- Performance metrics and analytics
- Save/load conversations
- Export conversations to various formats
- Plugin system for custom adapters
- Docker containers for each framework
- Kubernetes deployment configuration

## 12. References

- [AI Agent Frameworks Analysis](./AI-Agent-Frameworks-Analysis.md)
- [Claude Agent SDK Documentation](https://github.com/anthropics/anthropic-sdk-typescript)
- [LangChain Documentation](https://js.langchain.com/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [AutoGPT Documentation](https://docs.agpt.co/)

---

**Document Control**
- **Author**: System Architect
- **Last Updated**: January 15, 2026
- **Version**: 1.0
- **Status**: Draft → Review → Approved → Implementation
