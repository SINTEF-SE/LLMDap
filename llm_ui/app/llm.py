import transformers
import torch
import openai
import os
import sys

# Import API key
sys.path.append(os.path.dirname(__file__))
try:
    from openai_key import API_KEY
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    try:
        from profiler.openai_key import API_KEY
    except ImportError:
        API_KEY = None


class LLM:
    def __init__(self):
        # Set OpenAI API key
        if not API_KEY:
            raise ValueError("OpenAI API key not found. Please set it in llm_ui/app/openai_key.py or profiler/openai_key.py")
        
        self.client = openai.OpenAI(api_key=API_KEY)
        print("OpenAI client initialized successfully")
        
    def ask(self, prompt, max_tokens=500, temperature=0.3):
        """
        Send a prompt to OpenAI and get a response
        """
        try:
            print(f"Sending prompt to OpenAI (length: {len(prompt)} chars)")
            print(f"First 200 chars of prompt: {prompt[:200]}...")
            print(f"Last 200 chars of prompt: {prompt[-200:] if len(prompt) > 200 else prompt}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # You can change this to gpt-3.5-turbo if needed
                messages=[
                    {"role": "system", "content": "You are a helpful medical research assistant specializing in analyzing biomedical papers and datasets."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            raise e
