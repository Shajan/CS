/**
 * Calculator Tool
 * Performs basic arithmetic operations
 */

export interface CalculatorInput {
  operation: 'add' | 'subtract' | 'multiply' | 'divide';
  a: number;
  b: number;
}

export function calculator(input: CalculatorInput): number {
  const { operation, a, b } = input;

  switch (operation) {
    case 'add':
      return a + b;
    case 'subtract':
      return a - b;
    case 'multiply':
      return a * b;
    case 'divide':
      if (b === 0) {
        throw new Error('Cannot divide by zero');
      }
      return a / b;
    default:
      throw new Error(`Unknown operation: ${operation}`);
  }
}

export const calculatorToolDefinition = {
  name: 'calculator',
  description: 'Performs basic arithmetic operations (add, subtract, multiply, divide)',
  input_schema: {
    type: 'object',
    properties: {
      operation: {
        type: 'string',
        enum: ['add', 'subtract', 'multiply', 'divide'],
        description: 'The arithmetic operation to perform',
      },
      a: {
        type: 'number',
        description: 'The first number',
      },
      b: {
        type: 'number',
        description: 'The second number',
      },
    },
    required: ['operation', 'a', 'b'],
  },
};
