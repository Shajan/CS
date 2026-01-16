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
export declare function askUserQuestion(input: AskUserQuestionInput): {
    questions: Question[];
    note: string;
};
export declare const askUserQuestionToolDefinition: {
    name: string;
    description: string;
    input_schema: {
        type: string;
        properties: {
            questions: {
                type: string;
                description: string;
                items: {
                    type: string;
                    properties: {
                        question: {
                            type: string;
                            description: string;
                        };
                        header: {
                            type: string;
                            description: string;
                        };
                        options: {
                            type: string;
                            description: string;
                            items: {
                                type: string;
                                properties: {
                                    label: {
                                        type: string;
                                        description: string;
                                    };
                                    description: {
                                        type: string;
                                        description: string;
                                    };
                                };
                                required: string[];
                            };
                        };
                        multiSelect: {
                            type: string;
                            description: string;
                        };
                    };
                    required: string[];
                };
            };
        };
        required: string[];
    };
};
//# sourceMappingURL=ask-user-question.d.ts.map