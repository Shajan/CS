import unittest
from dotenv import load_dotenv
from llm import create_llm

class TestLLMs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        
    def test_list_models(self):
        model_configs = [
            ('openai', 'gpt-3.5-turbo'),
            ('claude', 'claude-2.1'),
            ('gemini', 'gemini-pro'),
            ('llama', 'llama-2-7b')
        ]
        
        for model_type, default_model in model_configs:
            with self.subTest(model_type=model_type):
                try:
                    llm = create_llm(model_type, default_model)
                    models = llm.list_available_models()
                    
                    print(f"\nAvailable {model_type} models:")
                    for model in models:
                        print(f"  - {model}")
                    
                    # Basic assertions
                    self.assertIsInstance(models, list)
                    self.assertTrue(len(models) > 0, f"No models found for {model_type}")
                    
                except Exception as e:
                    self.fail(f"Failed to test {model_type}: {str(e)}")

if __name__ == '__main__':
    unittest.main()
