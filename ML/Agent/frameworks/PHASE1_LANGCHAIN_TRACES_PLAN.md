# Phase 1: Fix LangChain Trace Emission

**Status**: 🔴 READY FOR IMPLEMENTATION
**Priority**: CRITICAL - Must be completed before adding new frameworks
**Estimated Duration**: 1 day
**Created**: 2026-01-15

---

## Problem Statement

The LangChain adapter currently **does NOT emit any traces**, making it impossible to:
- Compare performance metrics across frameworks
- Track token usage and costs
- Monitor tool usage
- Debug agent behavior
- Run fair benchmarks

This must be fixed before implementing OpenAI Agents SDK and Google ADK adapters to ensure all frameworks have parity in trace emission.

---

## Current State Analysis

### What Works
- ✅ LangChain adapter implements AgentAdapter interface
- ✅ Chat functionality works (sends/receives messages)
- ✅ Conversation history is maintained
- ✅ Model configuration is correct

### What's Missing
- ❌ No trace emission at all
- ❌ No USER_MESSAGE traces
- ❌ No API_REQUEST traces
- ❌ No API_RESPONSE traces (no token counts, no duration)
- ❌ No TOOL_CALL traces (even though LangChain supports tools)
- ❌ No TOOL_RESULT traces
- ❌ No ASSISTANT_RESPONSE traces
- ❌ No ERROR traces

---

## Reference Implementation (Claude Agent)

Claude Agent's ToolUsingAgent emits traces at these points:

```typescript
// 1. USER_MESSAGE - When user message is received
this.trace('USER_MESSAGE', {
  message: userMessage,
  conversationLength: this.conversationHistory.length,
});

// 2. API_REQUEST - Before calling API
this.trace('API_REQUEST', {
  model: this.model,
  max_tokens: this.maxTokens,
  messages: this.conversationHistory,
  tools: enabledTools,
  toolRound,
});

// 3. API_RESPONSE - After API responds
this.trace('API_RESPONSE', {
  id: response.id,
  model: response.model,
  stop_reason: response.stop_reason,
  usage: response.usage,
  duration_ms: duration,
  content: response.content,
  toolRound,
});

// 4. TOOL_CALL - When tool is called (if tools used)
this.trace('TOOL_CALL', {
  toolName: toolUse.name,
  toolInput: toolUse.input,
  toolUseId: toolUse.id,
  toolRound,
});

// 5. TOOL_RESULT - After tool execution (if tools used)
this.trace('TOOL_RESULT', {
  toolName: toolUse.name,
  toolUseId: toolUse.id,
  result,
  success: true,
  toolRound,
});

// 6. ASSISTANT_RESPONSE - Final response to user
this.trace('ASSISTANT_RESPONSE', {
  message: finalResponse,
  messageLength: finalResponse.length,
  tokensUsed: response.usage,
  toolRound,
});
```

---

## Implementation Plan

### Step 1: Import Dependencies

Add trace storage import at the top of `implementations/langchain/adapter.ts`:

```typescript
import { traceStorage } from '../../common/storage/trace-storage.js';
import type { TraceEvent, TraceCategory } from '../../common/types/trace.types.js';
```

### Step 2: Add emitTrace() Helper Method

Add this private method to the LangChainAdapter class:

```typescript
/**
 * Emit a trace event to the trace storage
 */
private emitTrace(sessionId: string, category: TraceCategory, data: any): void {
  const event: TraceEvent = {
    timestamp: new Date().toISOString(),
    sessionId,
    framework: this.name,
    category,
    data,
  };
  traceStorage.addTrace(event);
}
```

### Step 3: Update chat() Method - USER_MESSAGE

At the beginning of chat() method, after receiving the message, emit USER_MESSAGE:

```typescript
async chat(request: AgentRequest): Promise<AgentResponse> {
  const startTime = Date.now();

  // Emit USER_MESSAGE trace
  this.emitTrace(request.sessionId, 'USER_MESSAGE', {
    message: request.message,
  });

  // Get or create model for this session
  const model = this.getModel(request.sessionId, request.config);
  // ... rest of method
}
```

### Step 4: Update chat() Method - API_REQUEST

Before calling `model.invoke()`, emit API_REQUEST trace:

```typescript
// Add user message to history
const userMessage = new HumanMessage(request.message);
history.push(userMessage);

try {
  // Emit API_REQUEST trace
  this.emitTrace(request.sessionId, 'API_REQUEST', {
    model: request.config?.model || config.defaultConfig.model,
    max_tokens: request.config?.maxTokens || config.defaultConfig.maxTokens,
    temperature: request.config?.temperature ?? config.defaultConfig.temperature,
  });

  // Send messages to LangChain
  const apiStartTime = Date.now();
  const response = await model.invoke(history);
  const apiDuration = Date.now() - apiStartTime;
  // ... continue
}
```

### Step 5: Update chat() Method - API_RESPONSE

After receiving the response from LangChain, emit API_RESPONSE trace with token usage:

```typescript
const response = await model.invoke(history);
const apiDuration = Date.now() - apiStartTime;

// Extract usage information from LangChain response
// LangChain stores usage in response.response_metadata
const usage = (response as any).response_metadata?.usage || {};

// Emit API_RESPONSE trace
this.emitTrace(request.sessionId, 'API_RESPONSE', {
  id: (response as any).id || 'unknown',
  model: request.config?.model || config.defaultConfig.model,
  stop_reason: (response as any).response_metadata?.finish_reason || 'end_turn',
  duration_ms: apiDuration,
  usage: {
    input_tokens: usage.prompt_tokens || 0,
    output_tokens: usage.completion_tokens || 0,
  },
});
```

**Important Note**: OpenAI's response metadata structure:
- `response_metadata.usage.prompt_tokens` - input tokens
- `response_metadata.usage.completion_tokens` - output tokens
- `response_metadata.usage.total_tokens` - total tokens
- `response_metadata.finish_reason` - stop reason

### Step 6: Update chat() Method - ASSISTANT_RESPONSE

After the final response is assembled, emit ASSISTANT_RESPONSE trace:

```typescript
// Add assistant message to history
history.push(response);

const duration = Date.now() - startTime;

// Emit ASSISTANT_RESPONSE trace
this.emitTrace(request.sessionId, 'ASSISTANT_RESPONSE', {
  message: response.content as string,
});

return {
  response: response.content as string,
  sessionId: request.sessionId,
  framework: this.name,
  metadata: {
    model: config.defaultConfig.model,
    duration,
    usage: {
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: usage.completion_tokens || 0,
      total_tokens: usage.total_tokens || 0,
    },
  },
};
```

### Step 7: Add Error Handling

Wrap the API call in try-catch and emit ERROR trace on failure:

```typescript
try {
  // ... API call logic
} catch (error) {
  // Emit ERROR trace
  this.emitTrace(request.sessionId, 'ERROR', {
    message: error instanceof Error ? error.message : 'Unknown error',
    stack: error instanceof Error ? error.stack : undefined,
  });

  throw new Error(`LangChain chat failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
}
```

### Step 8: Tool Support (Optional - Future Enhancement)

LangChain supports tools through `bindTools()`. If tools are implemented later, add:

```typescript
// TOOL_CALL trace (when tool is invoked)
this.emitTrace(request.sessionId, 'TOOL_CALL', {
  toolName: toolCall.name,
  toolInput: toolCall.args,
  toolUseId: toolCall.id,
});

// TOOL_RESULT trace (after tool execution)
this.emitTrace(request.sessionId, 'TOOL_RESULT', {
  toolName: toolCall.name,
  toolUseId: toolCall.id,
  result: toolResult,
  success: !error,
  error: error?.message,
});
```

**Note**: This is NOT required for Phase 1. Tools can be added later.

---

## Complete Modified chat() Method Pseudocode

```typescript
async chat(request: AgentRequest): Promise<AgentResponse> {
  const startTime = Date.now();

  // 1. Emit USER_MESSAGE
  this.emitTrace(request.sessionId, 'USER_MESSAGE', {
    message: request.message,
  });

  // Get or create model for this session
  const model = this.getModel(request.sessionId, request.config);

  // Get conversation history
  const history = this.getHistoryMessages(request.sessionId);

  // Add user message to history
  const userMessage = new HumanMessage(request.message);
  history.push(userMessage);

  try {
    // 2. Emit API_REQUEST
    this.emitTrace(request.sessionId, 'API_REQUEST', {
      model: request.config?.model || config.defaultConfig.model,
      max_tokens: request.config?.maxTokens || config.defaultConfig.maxTokens,
      temperature: request.config?.temperature ?? config.defaultConfig.temperature,
    });

    // Send messages to LangChain
    const apiStartTime = Date.now();
    const response = await model.invoke(history);
    const apiDuration = Date.now() - apiStartTime;

    // Extract usage information
    const usage = (response as any).response_metadata?.usage || {};

    // 3. Emit API_RESPONSE
    this.emitTrace(request.sessionId, 'API_RESPONSE', {
      id: (response as any).id || 'unknown',
      model: request.config?.model || config.defaultConfig.model,
      stop_reason: (response as any).response_metadata?.finish_reason || 'end_turn',
      duration_ms: apiDuration,
      usage: {
        input_tokens: usage.prompt_tokens || 0,
        output_tokens: usage.completion_tokens || 0,
      },
    });

    // Add assistant message to history
    history.push(response);

    const duration = Date.now() - startTime;

    // 4. Emit ASSISTANT_RESPONSE
    this.emitTrace(request.sessionId, 'ASSISTANT_RESPONSE', {
      message: response.content as string,
    });

    return {
      response: response.content as string,
      sessionId: request.sessionId,
      framework: this.name,
      metadata: {
        model: config.defaultConfig.model,
        duration,
        usage: {
          input_tokens: usage.prompt_tokens || 0,
          output_tokens: usage.completion_tokens || 0,
          total_tokens: usage.total_tokens || 0,
        },
      },
    };
  } catch (error) {
    // 5. Emit ERROR
    this.emitTrace(request.sessionId, 'ERROR', {
      message: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : undefined,
    });

    throw new Error(`LangChain chat failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}
```

---

## Testing Strategy

### 1. Manual Testing

**Test case 1: Basic chat interaction**
```bash
# Start server
pnpm dev

# Use UI or CLI to send a message with LangChain
# Check browser console or network tab for traces
```

**Expected traces** (in order):
1. USER_MESSAGE - with message text
2. API_REQUEST - with model, max_tokens, temperature
3. API_RESPONSE - with id, model, stop_reason, duration_ms, usage (tokens)
4. ASSISTANT_RESPONSE - with response message

**Test case 2: Error handling**
```bash
# Remove OPENAI_API_KEY from .env
# Try to send a message
```

**Expected trace**:
- ERROR - with error message and stack

**Test case 3: Multi-turn conversation**
```bash
# Send 3 messages in a row
# Verify each interaction emits 4 traces
```

**Expected**:
- 3 × 4 = 12 total traces for the session

### 2. Trace Completeness Verification

**Check trace structure** in Smart View:
- ✅ All timestamps are sequential
- ✅ All sessionIds match
- ✅ All frameworks are 'langchain'
- ✅ All categories are valid (USER_MESSAGE, API_REQUEST, etc.)
- ✅ API_RESPONSE includes token counts
- ✅ API_RESPONSE includes duration_ms
- ✅ Duration is reasonable (500-3000ms typically)

**Compare with Claude Agent**:
- Both should emit same number of traces (4 per interaction without tools)
- Both should have similar data structures
- Both should include token usage

### 3. Token Accuracy Verification

**Manual check**:
1. Send a message: "What is 2+2?"
2. Check API_RESPONSE trace for usage data
3. Compare with OpenAI dashboard usage
4. Verify input_tokens and output_tokens are present
5. Verify total = input + output

**Common issues**:
- Missing usage data → Check response_metadata structure
- Tokens = 0 → LangChain might not be returning usage
- Wrong property names → OpenAI uses prompt_tokens/completion_tokens

### 4. Duration Accuracy Verification

**Check timing**:
- API_RESPONSE.duration_ms should be ~500-2000ms for typical requests
- Total metadata.duration should be >= API_RESPONSE.duration_ms
- Duration should not be 0

### 5. Unit Tests (Optional for Phase 1)

**Create test file**: `implementations/langchain/__tests__/trace-emission.test.ts`

```typescript
describe('LangChain Trace Emission', () => {
  it('should emit USER_MESSAGE trace', async () => {
    // Test implementation
  });

  it('should emit API_REQUEST trace', async () => {
    // Test implementation
  });

  it('should emit API_RESPONSE trace with tokens', async () => {
    // Test implementation
  });

  it('should emit ASSISTANT_RESPONSE trace', async () => {
    // Test implementation
  });

  it('should emit ERROR trace on failure', async () => {
    // Test implementation
  });
});
```

**Note**: Unit tests are good to have but NOT blocking for Phase 1 completion.

---

## Success Criteria

### Must Have (Blocking)
- ✅ LangChain adapter emits USER_MESSAGE traces
- ✅ LangChain adapter emits API_REQUEST traces
- ✅ LangChain adapter emits API_RESPONSE traces
- ✅ LangChain adapter emits ASSISTANT_RESPONSE traces
- ✅ API_RESPONSE includes token usage (input_tokens, output_tokens)
- ✅ API_RESPONSE includes duration_ms
- ✅ ERROR traces emitted on failures
- ✅ Token counts are accurate (verified against OpenAI dashboard)
- ✅ All traces have correct sessionId, framework, timestamp
- ✅ Traces appear in Smart View UI
- ✅ Traces appear in Raw JSON view

### Nice to Have (Non-blocking)
- ⭕ Unit tests for trace emission
- ⭕ Tool support (TOOL_CALL, TOOL_RESULT traces)
- ⭕ Detailed logging in development mode

---

## Files to Modify

### Primary File
- **`implementations/langchain/adapter.ts`** - Add trace emission

### No Changes Needed
- ❌ `common/storage/trace-storage.ts` - Already working
- ❌ `common/types/trace.types.ts` - Already has all trace types
- ❌ `implementations/langchain/config.json` - No changes needed

---

## Dependencies

### Already Available
- ✅ `traceStorage` from common/storage/trace-storage.ts
- ✅ `TraceEvent` interface from common/types/trace.types.ts
- ✅ `TraceCategory` type from common/types/trace.types.ts
- ✅ LangChain OpenAI integration (`@langchain/openai`)

### No New Dependencies Required
- No npm packages to install
- No new imports beyond what's already available

---

## Potential Issues & Solutions

### Issue 1: Token usage not available in response

**Symptoms**: `response_metadata.usage` is undefined

**Solutions**:
1. Check LangChain version (should be latest)
2. Check if model is configured correctly
3. Default to 0 if usage is missing (with console warning)
4. Update LangChain to latest version

**Code**:
```typescript
const usage = (response as any).response_metadata?.usage || {};
if (!usage.prompt_tokens) {
  console.warn('[LangChain] Token usage not available in response');
}
```

### Issue 2: Response structure changes

**Symptoms**: Can't access response.content or response_metadata

**Solutions**:
1. Log the full response to inspect structure
2. Check LangChain documentation for version differences
3. Add type guards and defaults

**Code**:
```typescript
console.log('[DEBUG] Full response:', JSON.stringify(response, null, 2));
```

### Issue 3: Trace not appearing in UI

**Symptoms**: Traces don't show up in Smart View or Raw JSON

**Solutions**:
1. Verify traceStorage.addTrace() is called
2. Check sessionId is correct
3. Check framework name matches 'langchain'
4. Verify traces API endpoint works

**Debugging**:
```typescript
console.log('[TRACE]', {
  sessionId: request.sessionId,
  framework: this.name,
  category,
  traceCount: traceStorage.getTraces(request.sessionId).length,
});
```

### Issue 4: Duration always 0

**Symptoms**: duration_ms is 0 or null

**Solutions**:
1. Verify Date.now() is called before and after API call
2. Check variable naming (apiStartTime vs startTime)
3. Ensure duration is calculated correctly

**Code**:
```typescript
const apiStartTime = Date.now();
const response = await model.invoke(history);
const apiDuration = Date.now() - apiStartTime;
console.log('[TIMING] API call took:', apiDuration, 'ms');
```

---

## Verification Checklist

Before marking Phase 1 as complete, verify:

- [ ] Code compiles without TypeScript errors
- [ ] Server starts without errors
- [ ] Can send message through UI with LangChain selected
- [ ] USER_MESSAGE trace appears in Smart View
- [ ] API_REQUEST trace appears in Smart View
- [ ] API_RESPONSE trace appears in Smart View
- [ ] ASSISTANT_RESPONSE trace appears in Smart View
- [ ] API_RESPONSE includes token counts (> 0)
- [ ] API_RESPONSE includes duration_ms (> 0)
- [ ] Timestamps are sequential
- [ ] Can export traces as JSON
- [ ] Can copy traces to clipboard
- [ ] Multi-turn conversations emit correct trace count
- [ ] Error handling emits ERROR trace
- [ ] All traces have correct sessionId
- [ ] All traces have framework = 'langchain'
- [ ] Token counts match OpenAI dashboard (approximately)

---

## Timeline

**Estimated time**: 4-6 hours

- **Setup & imports** (30 min) - Add imports, create emitTrace method
- **Implement trace emission** (2 hours) - Add all 4 trace points
- **Test basic functionality** (1 hour) - Send messages, verify traces appear
- **Token accuracy verification** (1 hour) - Check token counts, compare with OpenAI
- **Error handling** (30 min) - Test error cases
- **Multi-turn testing** (30 min) - Test conversation flows
- **Documentation** (30 min) - Update STATUS.md, commit changes

---

## Next Steps After Phase 1

Once LangChain trace emission is complete and verified:

1. ✅ Update STATUS.md - Mark Phase 1 complete
2. ✅ Commit changes with message: `feat(langchain): add comprehensive trace emission`
3. ✅ Push to origin
4. 🎯 Begin Phase 2 - OpenAI Agents SDK adapter
5. 🎯 Begin Phase 3 - Google ADK adapter (can parallel with Phase 2)

---

## References

### Internal Code References
- `implementations/claude-agent/agents/tool-using-agent.ts:33-178` - Reference implementation
- `implementations/claude-agent/agents/base-agent.ts:47-49` - trace() helper method
- `implementations/claude-agent/utils/trace-manager.ts:31-55` - Trace event creation
- `common/storage/trace-storage.ts:10-14` - addTrace() method
- `common/types/trace.types.ts:1-82` - All trace type definitions

### External Documentation
- [LangChain Response Metadata](https://js.langchain.com/docs/modules/model_io/models/chat/response_metadata)
- [OpenAI Token Usage](https://platform.openai.com/docs/guides/chat/response-format)
- [OpenAI Finish Reasons](https://platform.openai.com/docs/guides/chat/finish-reasons)

---

**Plan Status**: ✅ READY FOR IMPLEMENTATION
**Last Updated**: 2026-01-15
**Author**: Claude (Plan Mode)
**Reviewed**: Ready for user approval

---

**End of Phase 1 Plan**
