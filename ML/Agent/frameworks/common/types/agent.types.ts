/**
 * Request sent to any agent framework
 */
export interface AgentRequest {
  /** User message to send to the agent */
  message: string;

  /** Session identifier for conversation continuity */
  sessionId: string;

  /** Framework name (for routing) */
  framework?: string;

  /** Framework-specific configuration */
  config?: Record<string, any>;

  /** Request metadata */
  metadata?: Record<string, any>;
}

/**
 * Response from any agent framework
 */
export interface AgentResponse {
  /** Assistant's response message */
  response: string;

  /** Session identifier */
  sessionId: string;

  /** Framework that processed this request */
  framework: string;

  /** Response metadata */
  metadata?: {
    /** Model used for the response */
    model?: string;

    /** Number of tokens used */
    tokensUsed?: number;

    /** Response time in milliseconds */
    duration?: number;

    /** Framework-specific additional data */
    [key: string]: any;
  };
}

/**
 * Framework adapter capabilities
 */
export interface AdapterCapabilities {
  /** Supports streaming responses */
  supportsStreaming?: boolean;

  /** Supports tool/function calling */
  supportsTools?: boolean;

  /** Supports multimodal inputs (images, files, etc.) */
  supportsMultiModal?: boolean;

  /** Supports multiple cooperating agents */
  supportsMultiAgent?: boolean;

  /** Supports long-term memory */
  supportsMemory?: boolean;

  /** Maximum context length in tokens */
  maxContextLength?: number;

  /** List of supported models */
  supportedModels?: string[];
}

/**
 * Information about a registered adapter
 */
export interface AdapterInfo {
  /** Machine-readable name */
  name: string;

  /** Human-readable display name */
  displayName: string;

  /** Adapter version */
  version: string;

  /** Brief description */
  description?: string;

  /** Framework capabilities */
  capabilities?: AdapterCapabilities;

  /** Whether this adapter is enabled */
  enabled?: boolean;
}
