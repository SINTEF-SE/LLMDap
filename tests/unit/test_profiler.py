import unittest
import tempfile
import os
from unittest.mock import MagicMock, patch
import builtins

from llm_ui.app.pages import profiler
from profiler.metadata_schemas.arxpr2_schema import Metadata_form


class TestProfiler(unittest.TestCase):
    
    def test_handle_input_with_uploaded_file(self):
        mock_file = MagicMock()
        mock_file.name = "test.xml"
        mock_file.getvalue.return_value = b"<root></root>"

        path = profiler.handle_input(mock_file, None)
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"<root></root>")
    
    @patch('requests.get')
    def test_handle_input_with_url(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"<data>123</data>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        path = profiler.handle_input(None, "http://example.com/file.xml")
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"<data>123</data>")
    
    def test_handle_schema_with_file(self):
        mock_schema = MagicMock()
        mock_schema.name = "schema.json"
        mock_schema.getvalue.return_value = b'{"field": "value"}'
        
        path = profiler.handle_schema(mock_schema)
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b'{"field": "value"}')
        
    def test_handle_schema_fallback_default(self):
        result = profiler.handle_schema(None)
        self.assertEqual(result, Metadata_form)
        
    @patch("llm_ui.app.pages.profiler.call_inference")
    def test_run_pipeline_returns_cleaned_output(self, mock_call_inference):
        mock_call_inference.return_value = {
            "field1": {
                "context": {
                    "text": "Example.convertedFontNoiseHere"
                }
            }
        }

        dummy_xml = tempfile.NamedTemporaryFile(delete=False)
        dummy_xml.write(b"<xml></xml>")
        dummy_xml.close()

        class DummySchema:
            pass

        result = profiler.run_pipeline(dummy_xml.name, DummySchema())
        self.assertIn("field1", result)
        cleaned = result["field1"]["context"]["text"]
        self.assertNotIn("convertedFont", cleaned)

        os.unlink(dummy_xml.name)


if __name__ == "__main__":
    unittest.main()
        
        