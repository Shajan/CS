/**
 * Get Time Tool
 * Returns the current time in a specified timezone
 */
export interface GetTimeInput {
    timezone?: string;
}
export declare function getTime(input: GetTimeInput): string;
export declare const getTimeToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            timezone: {
                type: string;
                description: string;
            };
        };
        required: never[];
    };
};
//# sourceMappingURL=get-time.d.ts.map