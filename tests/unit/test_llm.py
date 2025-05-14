import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../llm_ui/app')))
from llm_ui.app.llm import LLM, load_settings

class TestLLM(unittest.TestCase):
    
    def setUp(self):
        self.settings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../llm_ui/app'))
        self.settings_file = "settings.json"
        self.settings_path = os.path.join(self.settings_dir, self.settings_file)
        
        os.makedirs(self.settings_dir, exist_ok=True)

        if os.path.exists(self.settings_file):
            os.remove(self.settings_file)
    
    def tearDown(self):
        self.settings_file = "settings.json"
        if os.path.exists(self.settings_file):
            os.remove(self.settings_file)
    
    def create_temp_settings_file(self, settings_data):
        with open(self.settings_file, 'w') as f:
            json.dump(settings_data, f)
    
    def test_llm_initialization_without_openai(self):
        mock_settings = {"use_openai": False, "model": "local_model"}
        mock_data = json.dumps(mock_settings)
    
        with patch("builtins.open", mock_open(read_data=mock_data)):
            llm = LLM()
        
            self.assertEqual(llm.settings['use_openai'], False)
            self.assertEqual(llm.settings['model'], 'local_model')
            self.assertEqual(llm.use_openai, False)
    
            
    @patch('openai.OpenAI')
    def test_llm_ask_openai(self, mock_openai):
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Test response"))])
        
        mock_settings = {"use_openai": True, "model": "gpt-4o", "temperature": 0.3, "max_tokens": 1450}
        self.create_temp_settings_file(mock_settings)
        llm = LLM()
        
        question = "What is the meaning of life?"
        response = llm.ask(question)
        
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {'role': 'system', 'content': 'You are a helpful medical research assistant specializing in analyzing biomedical papers and datasets.'},
                {'role': 'user', 'content': question}
            ],
            temperature=0.3,
            max_tokens=1450,
        )
        self.assertEqual(response, "Test response")
        del os.environ['OPENAI_API_KEY']
    
    @patch('llm_ui.app.llm.load_settings')
    def test_llm_ask_local(self, mock_load_settings):
        controlled_settings = {
            'temperature': 0.3,
            'max_tokens': 1450,
            'prompt_template': (
                "You are an AI assistant specializing in biomedical research datasets. Your task is to "
                "answer questions about the provided datasets from ArrayExpress/BioStudies.\n\n"
                "Title: {title}\nAbstract: {abstract}\n\nAvailable datasets:\n{datasets}\n\n"
                "Question: {question}\n\nPlease provide a comprehensive, accurate answer based on "
                "the dataset information provided above and make sure to include the PMID "
                " and/or the pubmed url link. \n"
            ),
            'model': "local_model",  # Set for this test
            'use_openai': False,     # Set for this test
            'similarity_k': 5,
            'profiler_options': {
                'field_info_to_compare': "choices"
            }
        }
        mock_load_settings.return_value = controlled_settings

    
        llm = LLM() 
            
        self.assertFalse(llm.settings.get("use_openai"), 
                         f"LLM.settings.use_openai should be False. Actual: {llm.settings}")
        self.assertEqual(llm.settings.get("model"), "local_model",
                         f"LLM.settings.model should be 'local_model'. Actual: {llm.settings}")
        
        self.assertIsNone(llm.client, "LLM.client should be None when use_openai is False.")
            
        question = "Is the local model implemented?"
        response = llm.ask(question)
        
        expected_response = "Error: Local model 'local_model' handling is not implemented."
        self.assertEqual(response, expected_response)
    
    @patch ('openai.OpenAI')
    def test_llm_ask_openai_error_handling(self, mock_openai):
        os.environ['OPENAI_API_KEY'] = 'test_api_key'
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        mock_settings = {"use_openai": True, "model": "gpt-3.5-turbo"}
        self.create_temp_settings_file(mock_settings)
        
        llm = LLM()
        question = "Will this throw an error?"
        
        response = llm.ask(question)  
        self.assertEqual(response, "An error occurred while communicating with the AI model: API error")  
        del os.environ['OPENAI_API_KEY']

if __name__ == '__main__':
    unittest.main()
            


    
    
        
        
    
    
        

