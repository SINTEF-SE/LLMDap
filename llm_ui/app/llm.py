import json
import openai
import os
import sys
# Note: Removed unused imports: transformers, torch

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

# Function to load settings from settings.json
# Duplicated here for simplicity, could be moved to a shared utils file
def load_settings():
    try:
        # Determine the correct path to settings.json relative to this script
        settings_path = os.path.join(os.path.dirname(__file__), '../../settings.json')
        # Adjust path if running from a different structure (e.g., within llm_ui/app)
        if not os.path.exists(settings_path):
             settings_path = os.path.join(os.path.dirname(__file__), '../../../settings.json') # Go up one more level

        # Fallback to current directory if not found yet
        if not os.path.exists(settings_path):
            settings_path = 'settings.json'

        if not os.path.exists(settings_path):
             print(f"Warning: settings.json not found at expected locations. Using defaults.")
             # Return defaults matching configure.py if file not found
             return {
                 'temperature': 0.3, 'max_tokens': 500,
                 'model': 'llama3.1I-8b-q4', 'use_openai': False
             }

        with open(settings_path, 'r') as f:
            settings_data = json.load(f)
            # Ensure essential keys exist, provide defaults if not
            settings_data.setdefault('model', 'llama3.1I-8b-q4')
            settings_data.setdefault('use_openai', False)
            settings_data.setdefault('temperature', 0.3)
            settings_data.setdefault('max_tokens', 500)
            return settings_data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading settings.json: {e}. Using defaults.")
        # Return defaults matching configure.py in case of error
        return {
            'temperature': 0.3, 'max_tokens': 500,
            'model': 'llama3.1I-8b-q4', 'use_openai': False
        }


class LLM:
    def __init__(self):
        # Load settings on initialization
        self.settings = load_settings()
        self.model = self.settings.get('model', '4o') # Default to '4o' if not found
        self.use_openai = self.settings.get('use_openai', True) # Default to True if not found

        # Set OpenAI API key only if using OpenAI
        if self.use_openai:
            if not API_KEY:
                raise ValueError("OpenAI API key not found, but use_openai is True. Please set the key.")
            self.client = openai.OpenAI(api_key=API_KEY)
            print(f"OpenAI client initialized successfully for model: {self.model}")
        else:
            # Placeholder for local model initialization if needed in the future
            self.client = None
            print(f"Configured for local model: {self.model} (OpenAI client not initialized)")

    def ask(self, prompt, max_tokens=None, temperature=None):
        """
        Send a prompt to the configured LLM (OpenAI or local) and get a response.
        Uses temperature/max_tokens from settings if not provided as arguments.
        """
        # Use provided args or fall back to loaded settings
        current_max_tokens = max_tokens if max_tokens is not None else self.settings.get('max_tokens', 500)
        current_temperature = temperature if temperature is not None else self.settings.get('temperature', 0.3)

        if not self.use_openai:
            # Placeholder for local model logic
            print(f"Warning: Local model ({self.model}) interaction not implemented yet.")
            return f"Error: Local model '{self.model}' handling is not implemented."

        if not self.client:
             return "Error: OpenAI client not initialized."

        try:
            print(f"Sending prompt to OpenAI model '{self.model}' (length: {len(prompt)} chars)")
            print(f"Params: max_tokens={current_max_tokens}, temperature={current_temperature}")
            # print(f"First 100 chars: {prompt[:100]}...") # Keep logging concise
            # print(f"Last 100 chars: ...{prompt[-100:] if len(prompt) > 100 else prompt}")

            # Map simplified model names from settings to actual OpenAI model IDs if necessary
            model_id = self.model
            if self.model == "4o":
                model_id = "gpt-4o"
            elif self.model == "4om":
                 model_id = "gpt-4o-mini"
            # Add other mappings if needed

            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful medical research assistant specializing in analyzing biomedical papers and datasets."}, # System prompt could also be configurable
                    {"role": "user", "content": prompt}
                ],
                max_tokens=current_max_tokens,
                temperature=current_temperature
            )
            print(f"Received response from {model_id}")
            return response.choices[0].message.content
        except openai.NotFoundError as e:
             print(f"Error: OpenAI model '{model_id}' not found or not accessible with your API key. {e}")
             return f"Error: The selected model '{model_id}' could not be found or accessed. Please check the model name and your API key permissions."
        except Exception as e:
            print(f"Error during OpenAI API call: {str(e)}")
            # Consider returning a user-friendly error message instead of raising
            return f"An error occurred while communicating with the AI model: {str(e)}"
