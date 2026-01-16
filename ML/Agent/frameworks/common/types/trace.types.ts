/**
 * Trace event types for monitoring agent execution
 */

export interface TraceEvent {
  timestamp: string;
  sessionId: string;
  framework: string;
  category: TraceCategory;
  data: TraceData;
}

export type TraceCategory =
  | 'USER_MESSAGE'
  | 'API_REQUEST'
  | 'API_RESPONSE'
  | 'ASSISTANT_RESPONSE'
  | 'TOOL_CALL'
  | 'TOOL_RESULT'
  | 'ERROR';

export type TraceData =
  | UserMessageData
  | ApiRequestData
  | ApiResponseData
  | AssistantResponseData
  | ToolCallData
  | ToolResultData
  | ErrorData;

export interface UserMessageData {
  message: string;
}

export interface ApiRequestData {
  model: string;
  max_tokens: number;
  temperature?: number;
  toolRound?: number;
}

export interface ApiResponseData {
  id: string;
  model: string;
  stop_reason: string;
  duration_ms: number;
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
}

export interface AssistantResponseData {
  message: string;
}

export interface ToolCallData {
  toolName: string;
  toolInput: any;
  toolUseId: string;
}

export interface ToolResultData {
  toolName: string;
  toolUseId: string;
  result: any;
  success: boolean;
  error?: string;
}

export interface ErrorData {
  message: string;
  stack?: string;
}

export interface TraceStorage {
  addTrace(trace: TraceEvent): void;
  getTraces(sessionId: string, framework?: string): TraceEvent[];
  clearTraces(sessionId: string, framework?: string): void;
  getAllSessions(): string[];
}
