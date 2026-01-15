/**
 * Get Time Tool
 * Returns the current time in a specified timezone
 */

export interface GetTimeInput {
  timezone?: string;
}

export function getTime(input: GetTimeInput): string {
  const { timezone = 'UTC' } = input;

  try {
    const now = new Date();
    const timeString = now.toLocaleString('en-US', {
      timeZone: timezone,
      dateStyle: 'full',
      timeStyle: 'long',
    });
    return timeString;
  } catch (error) {
    return `Error: Invalid timezone "${timezone}". Please use a valid IANA timezone (e.g., "America/New_York", "Europe/London", "Asia/Tokyo")`;
  }
}

export const getTimeToolDefinition = {
  name: 'get_time',
  description: 'Gets the current date and time in a specified timezone. Returns formatted datetime string.',
  input_schema: {
    type: 'object',
    properties: {
      timezone: {
        type: 'string',
        description: 'The IANA timezone name (e.g., "America/New_York", "Europe/London", "Asia/Tokyo"). Defaults to UTC if not specified.',
      },
    },
    required: [],
  },
};
