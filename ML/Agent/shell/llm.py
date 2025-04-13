from abc import ABC, abstractmethod
import os
import openai
import anthropic
import google.generativeai as genai

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.name = model_name
        self.model = self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model"""
        pass

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Return list of available models - same as supported models"""
        return cls.supported_models()

    @staticmethod
    @abstractmethod
    def supported_models() -> list[str]:
        """Return list of supported models for this type"""
        pass

    @staticmethod
    @abstractmethod
    def default_model() -> str:
        """Return default model name"""
        pass

class OpenAIModel(BaseModel):
    @staticmethod
    def supported_models() -> list[str]:
        return [
            'gpt-4-turbo-preview',
            'gpt-4',
            'gpt-4-0613',
            'gpt-4-0314',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-0613',
            'gpt-3.5-turbo-0301'
        ]

    @staticmethod
    def default_model() -> str:
        return "gpt-3.5-turbo"

    def _initialize_model(self):
        print(f"Initializing OpenAI model: {self.name}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self._llm = openai.OpenAI(api_key=api_key)
        return {"model_name": self.name, "type": "openai"}

class ClaudeModel(BaseModel):
    @staticmethod
    def supported_models() -> list[str]:
        return [
            'claude-2.1',
            'claude-2.0',
            'claude-instant-1.2'
        ]

    @staticmethod
    def default_model() -> str:
        return "claude-2.1"

    def _initialize_model(self):
        print(f"Initializing Claude model: {self.name}")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self._llm = anthropic.Anthropic(api_key=api_key)
        return {"model_name": self.name, "type": "claude"}

class GeminiModel(BaseModel):
    @staticmethod
    def supported_models() -> list[str]:
        return [
            'gemini-pro',
            'gemini-pro-vision'
        ]

    @staticmethod
    def default_model() -> str:
        return "gemini-pro"

    def _initialize_model(self):
        print(f"Initializing Gemini model: {self.name}")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self._llm = genai
        return {"model_name": self.name, "type": "gemini"}

class LlamaModel(BaseModel):
    @staticmethod
    def supported_models() -> list[str]:
        return [
            'llama-2-7b',
            'llama-2-13b',
            'llama-2-70b',
            'llama-2-7b-chat',
            'llama-2-13b-chat',
            'llama-2-70b-chat'
        ]

    @staticmethod
    def default_model() -> str:
        return "llama-2-7b"

    def _initialize_model(self):
        print(f"Initializing Llama model: {self.name}")
        return {"model_name": self.name, "type": "llama"}

_MODEL_CLASSES = {
    "openai": OpenAIModel,
    "claude": ClaudeModel,
    "gemini": GeminiModel,
    "llama": LlamaModel
}

def get_default_model(model_type: str) -> str:
    """Get default model name for a given model type"""
    if model_type not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model type: {model_type}")
    return _MODEL_CLASSES[model_type].default_model()

def create_llm(model_type: str, model_name: str = None) -> BaseModel:
    """Factory function to create model instances"""
    if model_type not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = _MODEL_CLASSES[model_type]
    if model_name is None:
        model_name = model_class.default_model()
    
    if model_name not in model_class.supported_models():
        raise ValueError(f"Unsupported model: {model_name}")
        
    return model_class(model_name)
