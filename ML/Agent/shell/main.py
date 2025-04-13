import os
import argparse
from dotenv import load_dotenv
from llm_factory import create_llm

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Create argument parser
    parser = argparse.ArgumentParser(description='Initialize and run LLM models')
    parser.add_argument('--model-type', type=str, default='openai',
                      choices=['openai', 'claude', 'gemini', 'llama'],
                      help='Type of LLM model to use (default: openai)')
    parser.add_argument('--model-name', type=str, default='gpt-3.5-turbo',
                      help='Specific model name (e.g., gpt-4-turbo-preview, gpt-3.5-turbo) (default: gpt-3.5-turbo)')
    parser.add_argument('--list-models', action='store_true',
                      help='List available models for the specified model type')

    # Parse arguments
    args = parser.parse_args()

    # Validate OpenAI model names
    valid_openai_models = ['gpt-4-turbo-preview', 'gpt-4', 'gpt-4-0613', 'gpt-4-0314', 
                          'gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0301']
    if args.model_type == 'openai' and args.model_name not in valid_openai_models:
        print(f"Error: Invalid OpenAI model name. Please choose from: {', '.join(valid_openai_models)}")
        return 1

    try:
        # Create LLM instance
        llm = create_llm(args.model_type, args.model_name)
        
        if args.list_models:
            models = llm.list_available_models()
            print(f"Available {args.model_type} models:")
            for model in models:
                print(f"  - {model}")
            return 0
            
        # TODO: Add your LLM interaction code here
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())