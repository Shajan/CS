/**
 * Agent configuration
 */
export interface AgentConfig {
  /** API key for the framework */
  apiKey?: string;

  /** Model to use */
  model?: string;

  /** Maximum tokens in response */
  maxTokens?: number;

  /** Temperature for response generation */
  temperature?: number;

  /** Session ID */
  sessionId?: string;

  /** Additional framework-specific configuration */
  [key: string]: any;
}

/**
 * Framework configuration metadata
 */
export interface FrameworkConfig {
  /** Machine-readable name */
  name: string;

  /** Human-readable display name */
  displayName: string;

  /** Version */
  version: string;

  /** Description */
  description?: string;

  /** Whether framework is enabled */
  enabled: boolean;

  /** Whether framework requires API key */
  requiresApiKey: boolean;

  /** Required environment variables */
  envVars: string[];

  /** Default configuration values */
  defaultConfig?: Record<string, any>;
}
