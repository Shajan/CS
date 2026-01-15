/**
 * WebFetch Tool
 * Fetch and parse web page content
 */

import https from 'https';
import http from 'http';

export interface WebFetchInput {
  url: string;
  prompt: string;
}

export function webFetch(input: WebFetchInput): Promise<{ content: string; url: string }> {
  const { url, prompt } = input;

  return new Promise((resolve, reject) => {
    try {
      const urlObj = new URL(url);
      const client = urlObj.protocol === 'https:' ? https : http;

      const request = client.get(url, (response) => {
        let data = '';

        response.on('data', (chunk) => {
          data += chunk;
        });

        response.on('end', () => {
          // Basic HTML to text conversion
          // In a real implementation, you would use a proper HTML parser
          const text = data
            .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
            .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
            .replace(/<[^>]+>/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

          resolve({
            content: text.substring(0, 10000), // Limit to first 10k chars
            url,
          });
        });
      });

      request.on('error', (error) => {
        reject(new Error(`Failed to fetch URL: ${error.message}`));
      });

      request.setTimeout(10000, () => {
        request.destroy();
        reject(new Error('Request timeout'));
      });
    } catch (error) {
      reject(new Error(`Invalid URL or fetch failed: ${error instanceof Error ? error.message : 'Unknown error'}`));
    }
  });
}

export const webFetchToolDefinition = {
  name: 'web_fetch',
  description: 'Fetches content from a URL and processes it. Returns the page content as text. Use the prompt parameter to specify what information to extract.',
  input_schema: {
    type: 'object',
    properties: {
      url: {
        type: 'string',
        description: 'The URL to fetch content from',
      },
      prompt: {
        type: 'string',
        description: 'Description of what information to extract from the page',
      },
    },
    required: ['url', 'prompt'],
  },
};
