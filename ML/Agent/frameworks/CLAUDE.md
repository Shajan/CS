# CLAUDE.md - AI Assistant Context & Guidance

**Purpose**: This document provides context and guidance for AI assistants (like Claude Code) working on this project.

**Last Updated**: January 15, 2026
**Project**: Multi-Framework AI Agent System

---

## Project Overview

This is a multi-framework AI agent testing system that allows comparison of different agent frameworks (Claude Agent SDK, LangChain, CrewAI, AutoGen, LlamaIndex, AutoGPT) through a unified interface.

**Current Status**: Phase 0 - Planning & Design (90% complete)
**See**: [STATUS.md](./STATUS.md) for detailed progress tracking

---

## Key Documents (Read These First!)

1. **[SPECIFICATION.md](./SPECIFICATION.md)** - Complete technical specification
   - Architecture design
   - API specifications
   - CLI specifications
   - All interfaces and types

2. **[STATUS.md](./STATUS.md)** - Project progress tracking
   - Phase-by-phase task lists
   - Completion status
   - Timeline and milestones

3. **[AI-Agent-Frameworks-Analysis.md](./AI-Agent-Frameworks-Analysis.md)** - Framework research
   - Market overview
   - Framework comparison
   - Use cases and recommendations

4. **This file (CLAUDE.md)** - Context for AI assistants

---

## Project Structure

```
frameworks/
├── SPECIFICATION.md              # Technical spec (READ THIS!)
├── STATUS.md                     # Progress tracker (UPDATE THIS!)
├── CLAUDE.md                     # This file
├── AI-Agent-Frameworks-Analysis.md
│
├── common/                       # Shared code (TO BE CREATED)
│   ├── types/                   # TypeScript interfaces
│   ├── api/                     # Framework-agnostic API server
│   └── adapters/                # Adapter infrastructure
│
├── implementations/              # Framework-specific code (TO BE CREATED)
│   ├── claude-agent/            # Will be migrated from claude.agent/
│   ├── langchain/               # To be implemented
│   ├── crewai/                  # To be implemented
│   └── ...
│
├── ui/                          # React web interface (TO BE CREATED)
│   └── src/
│       ├── components/
│       └── App.tsx
│
├── cli/                         # Command-line interface (TO BE CREATED)
│   └── src/
│       ├── commands/
│       └── index.ts
│
└── claude.agent/                # EXISTING implementation (will be refactored)
    ├── server/                  # Express server + agents
    │   ├── agents/             # Agent implementations
    │   │   ├── base-agent.ts
    │   │   ├── simple-chat-agent.ts
    │   │   └── tool-using-agent.ts
    │   ├── routes/             # API routes
    │   ├── tools/              # Tool implementations
    │   └── index.ts
    └── src/                     # React UI
        ├── App.tsx
        ├── Chat.tsx
        ├── Settings.tsx
        └── Traces.tsx
```

---

## Important Files & Their Purpose

### Planning Documents (Current Focus)
- `SPECIFICATION.md` - The blueprint for everything
- `STATUS.md` - Track what's done, what's next
- `CLAUDE.md` - This file

### Existing Implementation (To Be Refactored)
- `claude.agent/server/agents/base-agent.ts` - Base agent class (good reference for adapter pattern)
- `claude.agent/server/routes/agent.ts` - Current API routes (will be replaced)
- `claude.agent/src/Chat.tsx` - Current chat UI (will be enhanced)

### To Be Created (Phase 1-6)
- `common/adapters/adapter.interface.ts` - Core abstraction (CRITICAL)
- `common/adapters/adapter-registry.ts` - Framework routing (CRITICAL)
- `common/api/server.ts` - New unified API server
- `implementations/claude-agent/adapter.ts` - First adapter implementation

---

## Available Tools & Capabilities

### Standard Development Tools
- **Read**: Read files (use liberally to understand code)
- **Write**: Create new files (use for new implementations)
- **Edit**: Modify existing files (use for refactoring)
- **Glob**: Find files by pattern
- **Grep**: Search file contents
- **Bash**: Run commands (git, npm, pnpm, build, test)

### Browser Automation (Playwright MCP Server)
You have access to a **Playwright MCP server** for browser automation and testing!

**Available commands** (prefix: `mcp__playwright__`):
- `browser_navigate` - Navigate to URL
- `browser_snapshot` - Capture accessibility snapshot (better than screenshot)
- `browser_take_screenshot` - Take visual screenshots
- `browser_click` - Click elements
- `browser_type` - Type into inputs
- `browser_fill_form` - Fill multiple form fields
- `browser_evaluate` - Run JavaScript in browser
- `browser_console_messages` - Get console output
- `browser_network_requests` - Inspect network activity
- `browser_wait_for` - Wait for elements or text

**Use cases for this project**:
- Test the React UI in a real browser
- Verify framework selection dropdown works
- Test chat interactions end-to-end
- Capture screenshots for documentation
- Debug UI issues visually
- Test framework switching behavior

**Example workflow**:
```typescript
// 1. Start the dev server first
// Bash: pnpm dev

// 2. Navigate to UI
mcp__playwright__browser_navigate({ url: "http://localhost:5173" })

// 3. Take snapshot to see what's on page
mcp__playwright__browser_snapshot({})

// 4. Test framework selector
mcp__playwright__browser_click({ element: "framework dropdown", ref: "..." })

// 5. Verify chat works
mcp__playwright__browser_fill_form({ fields: [...] })
```

---

## Development Workflows

### Starting a New Phase

1. **Read STATUS.md** to see current phase and tasks
2. **Read SPECIFICATION.md** section for that phase
3. **Create task list** using TodoWrite tool
4. **Implement tasks** one by one
5. **Update STATUS.md** - check off completed tasks
6. **Run tests** to verify
7. **Commit changes** with descriptive messages

### Adding a New Framework Adapter

1. **Read** `SPECIFICATION.md` Section 3.1 (AgentAdapter Interface)
2. **Create** `implementations/<framework>/` directory
3. **Implement** adapter.ts with AgentAdapter interface
4. **Create** config.json with framework metadata
5. **Write tests** for the adapter
6. **Register** adapter in server startup
7. **Update** STATUS.md Phase 6 section
8. **Document** setup instructions

### Testing UI Changes

1. **Start server**: `cd claude.agent && pnpm dev`
2. **Use Playwright** to navigate and interact
3. **Take snapshots** to verify UI state
4. **Check console** for errors
5. **Test framework selection** workflow
6. **Capture screenshots** for documentation

### Before Committing Code

- [ ] Run tests: `pnpm test`
- [ ] Check TypeScript: `pnpm build`
- [ ] Update STATUS.md checkboxes
- [ ] Update CLAUDE.md if structure changed
- [ ] Write descriptive commit message

---

## Coding Conventions

### TypeScript
- Use **interfaces** for public APIs and contracts
- Use **types** for unions, intersections, utilities
- Enable **strict mode** (already configured)
- Prefer **readonly** for interface properties that shouldn't change
- Export interfaces with clear names: `AgentAdapter`, `AgentRequest`, etc.

### File Naming
- **Interfaces**: `agent.interface.ts`, `adapter.interface.ts`
- **Types**: `agent.types.ts`, `message.types.ts`
- **Implementations**: `adapter.ts`, `registry.ts`
- **Tests**: `*.test.ts`, `*.spec.ts`
- **Components**: `PascalCase.tsx` (React)

### Code Organization
- **One main export per file** (makes imports cleaner)
- **Co-locate related files** (adapters together, routes together)
- **Separate concerns** (types, logic, UI, API)
- **Use index.ts** to re-export from directories

### Comments
- **Interface methods**: Document with JSDoc
- **Complex logic**: Explain WHY, not WHAT
- **TODOs**: Use `// TODO(phase-N): Description`
- **Framework-specific**: Note peculiarities or workarounds

### Git Commits
- Format: `<type>: <description>`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Examples:
  - `feat(adapters): implement AgentAdapter interface`
  - `refactor(claude-agent): move to implementations directory`
  - `docs: update STATUS.md Phase 1 progress`

---

## Common Tasks

### Read Current Implementation
```typescript
// Start by understanding the existing code
Read("claude.agent/server/agents/base-agent.ts")
Read("claude.agent/server/routes/agent.ts")
Read("claude.agent/src/Chat.tsx")
```

### Create Core Types (Phase 1)
```typescript
// Create the foundation
Write("common/types/agent.types.ts", { content: "..." })
Write("common/adapters/adapter.interface.ts", { content: "..." })
```

### Test with Playwright
```bash
# Start dev server
cd claude.agent && pnpm dev

# Then use Playwright tools to test
mcp__playwright__browser_navigate({ url: "http://localhost:5173" })
mcp__playwright__browser_snapshot({})
```

### Update Progress
```typescript
// After completing tasks, update STATUS.md
Edit("STATUS.md", {
  old_string: "- [ ] Create adapter interface",
  new_string: "- [x] Create adapter interface"
})
```

### Run Tests
```bash
# Unit tests
pnpm test

# Type checking
pnpm build

# Specific test file
pnpm test adapter.test.ts
```

---

## Key Architectural Concepts

### Adapter Pattern (Core Abstraction)
Every framework implements the **AgentAdapter interface**:
```typescript
interface AgentAdapter {
  chat(request: AgentRequest): Promise<AgentResponse>
  clearHistory(sessionId: string): Promise<void>
  getHistory(sessionId: string): Promise<ConversationHistory>
}
```

**Why?** Allows API to be completely framework-agnostic.

### Registry Pattern (Framework Routing)
The **AdapterRegistry** maps framework names to adapters:
```typescript
registry.register(new ClaudeAgentAdapter())
registry.getAdapter("claude-agent") // Returns ClaudeAgentAdapter
```

**Why?** Single place to discover and route to frameworks.

### Separation of Concerns
- **UI**: Framework selection, chat interface (no framework logic)
- **API**: Route requests, validate, respond (no framework logic)
- **Adapters**: Framework-specific implementation (isolated)

**Why?** Each part can change independently.

---

## Environment & Dependencies

### Environment Variables
Create `.env` in project root:
```bash
# Claude Agent SDK
ANTHROPIC_API_KEY=sk-ant-xxx

# LangChain
OPENAI_API_KEY=sk-xxx
LANGCHAIN_API_KEY=xxx

# Other frameworks (add as needed)
CREWAI_API_KEY=xxx
AUTOGEN_API_KEY=xxx
```

### Package Manager
- **Use pnpm** (not npm or yarn)
- Workspaces configured in root `package.json`
- Commands: `pnpm install`, `pnpm dev`, `pnpm test`, `pnpm build`

### Tech Stack
- **Backend**: Node.js, Express, TypeScript
- **Frontend**: React, Vite, TypeScript
- **CLI**: Commander, TypeScript
- **Testing**: Jest (to be configured)
- **Frameworks**: Various (Anthropic SDK, LangChain, etc.)

---

## Troubleshooting

### Can't Find Files
- Check if they exist: `Glob("**/*.ts")`
- Remember: Many files don't exist yet (see STATUS.md)
- Current implementation is in `claude.agent/`

### TypeScript Errors
- Check `tsconfig.json` for correct paths
- Ensure types are exported properly
- Run `pnpm build` to see all errors

### API Not Working
- Check server is running: `pnpm dev`
- Check correct port: `http://localhost:3001`
- Check environment variables loaded
- Check adapter is registered

### UI Not Showing Framework Selector
- Feature not implemented yet (Phase 4)
- See STATUS.md for current phase
- Can test with Playwright once implemented

### Framework Adapter Failing
- Check framework is installed: `pnpm install`
- Check API key is set in `.env`
- Check adapter implements all required methods
- Check adapter is registered on server startup

---

## Testing Strategy

### Unit Tests
- Test adapters in isolation with mocks
- Test registry with mock adapters
- Test API routes with mock adapters
- Test CLI commands with mock API

### Integration Tests
- Test API + real adapter
- Test UI + API + adapter
- Test CLI + API + adapter

### End-to-End Tests (with Playwright)
1. Start server: `pnpm dev`
2. Navigate to UI: `browser_navigate`
3. Select framework: `browser_click`
4. Send message: `browser_fill_form`
5. Verify response: `browser_snapshot`
6. Check traces: Navigate to traces view

### Manual Testing Checklist
- [ ] Can select framework from dropdown
- [ ] Can send message and get response
- [ ] Can switch frameworks (clears history)
- [ ] Can view traces
- [ ] Can configure settings
- [ ] CLI chat works
- [ ] CLI test works
- [ ] CLI benchmark works

---

## When Starting a New Session

1. **Read STATUS.md first** to see where we are
2. **Check the current phase** and what's next
3. **Review recent changes**: `git log --oneline -10`
4. **Check working tree**: `git status`
5. **Read relevant SPECIFICATION.md section** for context
6. **Create TodoWrite list** for your session goals
7. **Start implementing** following the conventions above

---

## Important Reminders

### Do's ✅
- Read STATUS.md before starting work
- Update STATUS.md after completing tasks
- Use TodoWrite to track session progress
- Follow the adapter pattern strictly
- Test with Playwright for UI changes
- Write tests for new code
- Keep framework implementations isolated
- Document framework-specific quirks
- Use TypeScript strict mode
- Follow file naming conventions

### Don'ts ❌
- Don't skip reading SPECIFICATION.md
- Don't forget to update STATUS.md
- Don't mix framework logic in API layer
- Don't hardcode framework names in UI/API
- Don't forget to test with multiple frameworks
- Don't commit API keys or secrets
- Don't change interfaces without updating all adapters
- Don't break existing claude.agent functionality

---

## Quick Reference Commands

```bash
# Development
pnpm install                    # Install dependencies
pnpm dev                        # Start dev server
pnpm build                      # Build for production
pnpm test                       # Run tests

# Git
git status                      # Check working tree
git log --oneline -10          # Recent commits
git diff                        # See changes

# Project Navigation
ls -la claude.agent/server/    # Existing implementation
ls -la common/                  # New shared code (TBD)
ls -la implementations/         # Framework adapters (TBD)

# Testing with Playwright
# 1. Start server: pnpm dev
# 2. Use mcp__playwright__* tools to interact with UI
```

---

## Resources & Links

### Internal Documentation
- [SPECIFICATION.md](./SPECIFICATION.md) - Technical spec
- [STATUS.md](./STATUS.md) - Progress tracking
- [AI-Agent-Frameworks-Analysis.md](./AI-Agent-Frameworks-Analysis.md) - Research

### External Documentation
- [Claude Agent SDK](https://github.com/anthropics/anthropic-sdk-typescript)
- [LangChain JS](https://js.langchain.com/)
- [CrewAI](https://docs.crewai.com/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [AutoGPT](https://docs.agpt.co/)

### Tools
- [Playwright Docs](https://playwright.dev/)
- [Commander.js](https://github.com/tj/commander.js)
- [Express](https://expressjs.com/)
- [Vite](https://vitejs.dev/)

---

## Session Context Template

When starting a new session, Claude Code should note:

**Current Phase**: [Phase name from STATUS.md]
**Current Tasks**: [List of active tasks]
**Blockers**: [Any blockers or dependencies]
**Today's Goal**: [What to accomplish this session]

---

## Contact & Feedback

If this document is unclear or missing important information:
- Update this file with clarifications
- Add new sections as needed
- Keep it concise but comprehensive
- Date all significant updates

---

**Last Updated**: January 15, 2026
**Maintained By**: Project Team & AI Assistants
**Status**: Living Document - Update as project evolves

---

## Appendix: File Creation Checklist

When creating a new file, ensure:
- [ ] Proper TypeScript types/interfaces
- [ ] JSDoc comments for public APIs
- [ ] Tests file created (*.test.ts)
- [ ] Exports are clear and documented
- [ ] File follows naming conventions
- [ ] Related STATUS.md task checked off
- [ ] File added to relevant index.ts

When creating a new adapter:
- [ ] Implements AgentAdapter interface
- [ ] Has config.json with metadata
- [ ] Has tests with standard test suite
- [ ] Registered in AdapterRegistry
- [ ] Documented in README
- [ ] Added to STATUS.md Phase 6
- [ ] Environment variables documented

---

**End of CLAUDE.md**
