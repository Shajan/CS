/**
 * AskUserQuestion Tool
 * Ask the user clarifying questions with multiple choice options
 */

export interface QuestionOption {
  label: string;
  description: string;
}

export interface Question {
  question: string;
  header: string;
  options: QuestionOption[];
  multiSelect: boolean;
}

export interface AskUserQuestionInput {
  questions: Question[];
}

export function askUserQuestion(input: AskUserQuestionInput): { questions: Question[]; note: string } {
  const { questions } = input;

  // This tool requires interactive user input
  // In a real implementation, this would:
  // 1. Pause the agent execution
  // 2. Display questions to the user in the UI
  // 3. Wait for user responses
  // 4. Resume execution with the answers

  return {
    questions,
    note: 'This tool requires interactive UI implementation. Questions would be displayed to the user for input.',
  };
}

export const askUserQuestionToolDefinition = {
  name: 'ask_user_question',
  description: 'Ask the user questions during execution to gather preferences, clarify requirements, or get decisions on implementation choices. Supports single or multiple choice questions.',
  input_schema: {
    type: 'object',
    properties: {
      questions: {
        type: 'array',
        description: 'Questions to ask the user (1-4 questions)',
        items: {
          type: 'object',
          properties: {
            question: {
              type: 'string',
              description: 'The complete question to ask',
            },
            header: {
              type: 'string',
              description: 'Short label displayed as a tag (max 12 chars)',
            },
            options: {
              type: 'array',
              description: 'Available choices (2-4 options)',
              items: {
                type: 'object',
                properties: {
                  label: {
                    type: 'string',
                    description: 'Display text for this option',
                  },
                  description: {
                    type: 'string',
                    description: 'Explanation of what this option means',
                  },
                },
                required: ['label', 'description'],
              },
            },
            multiSelect: {
              type: 'boolean',
              description: 'Allow multiple options to be selected',
            },
          },
          required: ['question', 'header', 'options', 'multiSelect'],
        },
      },
    },
    required: ['questions'],
  },
};
