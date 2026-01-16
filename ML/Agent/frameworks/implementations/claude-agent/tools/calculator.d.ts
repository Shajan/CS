/**
 * Calculator Tool
 * Performs basic arithmetic operations
 */
export interface CalculatorInput {
    operation: 'add' | 'subtract' | 'multiply' | 'divide';
    a: number;
    b: number;
}
export declare function calculator(input: CalculatorInput): number;
export declare const calculatorToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            operation: {
                type: string;
                enum: string[];
                description: string;
            };
            a: {
                type: string;
                description: string;
            };
            b: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=calculator.d.ts.map