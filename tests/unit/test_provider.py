import unittest
import os
import tempfile
import shutil
import json
import sys
from unittest.mock import MagicMock, patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
mock_st_singleton = MagicMock() 
mock_st_experimental_singleton = MagicMock()

def passthrough_decorator(func_to_decorate=None, **kwargs):
    if func_to_decorate is None:
        return lambda f: f
    return func_to_decorate

mock_st_cache_resource = MagicMock(side_effect=passthrough_decorator)

STREAMLIT_MOCKS = {
    'error': MagicMock(),
    'info': MagicMock(),
    'warning': MagicMock(),
    'success': MagicMock(),
    'title': MagicMock(),
    'subheader': MagicMock(),
    'container': MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=None)),
    'radio': MagicMock(return_value="Default"),
    'file_uploader': MagicMock(),
    'text_input': MagicMock(return_value=""),
    'text_area': MagicMock(return_value=""),
    'button': MagicMock(return_value=False),
    'caption': MagicMock(),
    'markdown': MagicMock(),
    'spinner': MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=None)),
    'cache_resource': mock_st_cache_resource, 
    'cache_data': MagicMock(side_effect=passthrough_decorator), 
    'session_state': type('SessionState', (), {'get': MagicMock(return_value=None), '__contains__': MagicMock(return_value=False), '__setitem__': MagicMock(), '__getitem__': MagicMock(side_effect=KeyError("mock"))})(),
    'set_page_config': MagicMock(),
    'sidebar': MagicMock(),
    'expander': MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=None)),
    'columns': MagicMock(return_value=(MagicMock(), MagicMock())),
    'cache': MagicMock(side_effect=passthrough_decorator), 
}


patchers = []
for target, mock_obj in STREAMLIT_MOCKS.items():
    patchers.append(patch(f'streamlit.{target}', mock_obj))
    
def apply_class_patches(cls):
    for p in patchers:
        cls = p(cls)
    return cls

from llm_ui.app.pages import provider

@apply_class_patches
class TestProvider(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        for mock_obj in STREAMLIT_MOCKS.values():
            if hasattr(mock_obj, 'reset_mock'):
                mock_obj.reset_mock()
        
        if hasattr(STREAMLIT_MOCKS['session_state'], 'clear'):
            STREAMLIT_MOCKS['session_state'].clear()
        else:
            STREAMLIT_MOCKS['session_state'].get = MagicMock(return_value=None)
            STREAMLIT_MOCKS['session_state'].__contains__ = MagicMock(return_value=False)
            STREAMLIT_MOCKS['session_state'].__setitem__ = MagicMock()
            STREAMLIT_MOCKS['session_state'].__getitem__ = MagicMock(side_effect=KeyError("mock"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_try_extract_pmid_from_xml_article_id_tag(self):
        xml_content = """<?xml version="1.0" ?>
        <root>
            <article>
                <front>
                    <article-meta>
                        <article-id pub-id-type="other">10.1234/otherid</article-id>
                        <article-id pub-id-type="pmid">12345678</article-id>
                        <article-id pub-id-type="doi">10.1234/doiid</article-id>
                    </article-meta>
                </front>
            </article>
        </root>
        """
        
        file_path = os.path.join(self.temp_dir, "test_article.xml")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        
        extracted_pmid = provider._try_extract_pmid(file_path)
        
        self.assertEqual(extracted_pmid, "12345678")
    
    def test_try_extract_pmid_from_text_content_regex(self):
        text_content = """
        This is a sample text document.
        It contains a PubMed ID for reference.
        PMID: 11223344
        Some more text after the PMID.
        Also, PubMed ID : 55667788
        """
        
        file_path = os.path.join(self.temp_dir, "sample_text.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
            
        extracted_pmid = provider._try_extract_pmid(file_path)
        self.assertEqual(extracted_pmid, "11223344")
    
    def test_try_extract_pmid_with_no_pmid_present(self):
        content = """
        <root>
            <data>This XML file has no PMID.</data>
            <info>Nor does this text.</info>
        </root>
        """
        
        file_path = os.path.join(self.temp_dir, "no_pmid_xml")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        extracted_pmid = provider._try_extract_pmid(file_path)
        self.assertIsNone(extracted_pmid)
    
    def test_try_extract_pmid_file_not_exists(self):
        non_existent_file_path = os.path.join(self.temp_dir, "surely_non_existent_file.xml")
        if os.path.exists(non_existent_file_path):
            os.remove(non_existent_file_path)
        
        extracted_pmid = provider._try_extract_pmid(non_existent_file_path)
        self.assertIsNone(extracted_pmid)

if __name__ == '__main__':
    unittest.main(verbosity=2)