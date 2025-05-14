import unittest
from unittest.mock import patch, MagicMock
import os
import json
from llm_ui.app.llm import LLM, load_settings

class TestLLM(unittest.TestCase):
    
    def setUp(self):
        self.settings_file = "settings.json"
        if os.path.exists(self.settings_file):
            os.remove(self.settings_file)
    
    def tearDown(self):
        self.settings_file = "settings.json"
        if os.path.exists(self.settings_file):
            os.remove(self.settings_file)
    
    def create_temp_settings_file(self, settings_data):
        settings_path = self.settings_file
        with open(self.settings_file, 'w') as f:
            json.dump(settings_data, f)
    
    def test_load_settings_file_exists(self):
        mock_settings = {"model": "test_model", "temperature": 0.5, "max_tokens": 100, "use_openai" : False}
        self.create_temp_settings_file(mock_settings)
        
        loaded_settings = load_settings()
        
        self.assertEqual(loaded_settings, mock_settings)
    
    def test_load_settings_file_not_exists(self):
        loaded_settings = load_settings()
        
        default_settings = {
            'temperature': 0.3,
            'max_tokens': 500,
            'model': 'llama3.1I-8b-q4', 
            'use_openai': False
        }
        self.assertTrue(default_settings.items() <= loaded_settings.items())
    
    @patch('openai.OpenAI')
    def test_llm_initialization_with_openai(self, mock_openai):
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_settings = {"use_openai": True, "model": "gpt-3.5-turbo"}
        self.create_temp_settings_file(mock_settings)
        
        llm = LLM()
        
        mock_openai.assert_called_once_with(api_key='test_api_key')
        self.assertEqual(llm.client, mock_client)
        self.assertEqual(llm.settings, mock_settings)
        del os.environ['OPENAI_API_KEY']
    
    def test_llm_intialization_without_openai(self):
        mock_settings = {"use_openai": False, "model": "local_model"}
        self.create_temp_settings_file(mock_settings)
        llm = LLM()
        
        self.assertIsNone(llm.client)
        self.assertTrue(mock_settings.items() <= llm.settings.items())
    
    def test_llm_initialization_no_openai_key(self):
        mock_settings = {"use_openai": True, "model": "gpt-3.5-turbo"}
        self.create_temp_settings_file(mock_settings)
        with self.assertRaises(ValueError):
            LLM()
    
    @patch('openai.OpenAI')
    def test_llm_ask_openai(self, mock_openai):
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        print(f"API Key: {os.environ.get('OPENAI_API_KEY')}") #DEBUGGING KODE - SLETT
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_completion
        
        mock_settings = {"use_openai": True, "model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 200}
        self.create_temp_settings_file(mock_settings)
        llm = LLM()
        
        question = "What is the meaning of life?"
        response = llm.ask(question)
        
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=200,
        )
        self.assertEqual(response, "Test response")
        del os.environ['OPENAI_API_KEY']
        
    @patch ('openai.OpenAI')
    def test_llm_ask_openai_override_params(self, mock_openai):
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_completion
        
        mock_settings = {'use_openai': True, "model": "gpt-3.5-turbo", 'temperature': 0.7, "max_tokens": 200}
        self.create_temp_settings_file(mock_settings)
        
        llm = LLM()
        question = "What is your name?"
        response = llm.ask(question, temperature=0.9, max_tokens=300)
        
        mock_client.chat.completions.create.assert_called_once_with(
            model = "gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            temperature=0.9,
            max_tokens=300,
        )
        self.assertEqual(response, "Test response")
        del os.environ['OPENAI_API_KEY']
    
    def test_llm_ask_local(self):
        mock_settings = {"use_openai": False, "model": "local_model"}
        self.create_temp_settings_file(mock_settings)
        llm = LLM()
        print(f"LLM settings: {llm.settings}")
        question = "Is the local model implemented?"
        response = llm.ask(question)
        self.assertEqual(response, "Error: Local model 'local_model' handling is not implemented.")
    
    @patch ('openai.OpenAI')
    def test_llm_ask_openai_error_handling(self, mock_openai):
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        mock_settings = {"use_openai": True, "model": "gpt-3.5-turbo"}
        self.create_temp_settings_file(mock_settings)
        
        llm = LLM()
        question = "Will this throw an error?"
        
        with self.assertRaises(Exception) as context:
            llm.ask(question)
        self.assertEqual(str(context.exception), "API error")
        del os.environ['OPENAI_API_KEY']

if __name__ == '__main__':
    unittest.main()


