import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import builtins
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from llm_ui.app.pages.consumer_QA import extract_dataset_metadata

class TestConsumerQA(unittest.TestCase):
    
    def test_overview_section_basic_fields(self):
        dataset = {
            'title': 'Test Title',
            'source': 'user_provider',
            'pmid': '123456',
            'accession': 'E-TEST-1234',
            'organism': 'Homo sapiens',
            'study_type': 'Transcriptomics'
        }
        
        with patch('llm_ui.app.pages.consumer_QA._fetch_pubmed_context', return_value=["Article Title: Test", "Abstract: Some abstract"]), \
             patch('llm_ui.app.pages.consumer_QA._fetch_europepmc_context', return_value=[]), \
             patch('llm_ui.app.pages.consumer_QA._fetch_arrayexpress_context', return_value=[]), \
             patch('builtins.open', mock_open(read_data=json.dumps({"context": {"snippet_1": "A short snippet"}}))), \
             patch('os.path.exists', return_value=True):

            
            result = extract_dataset_metadata(dataset)
            self.assertIn("### Dataset Overview: Test Title", result)
            self.assertIn("**Fetched Publication Abstract:** Some abstract", result)
            
    def test_invalid_pmid_handling(self):
        dataset = {
            'title': 'No Pub Info',
            'source': 'arrayexpress',
            'pmid': 'NO_PMID',
            'accession': 'E-EXPR-0001'
        }
        
        with patch('llm_ui.app.pages.consumer_QA._fetch_arrayexpress_context', return_value=[]):
            result = extract_dataset_metadata(dataset)
            self.assertIn("- Publication Info: No valid PMID provided.", result)
            
    def test_missing_json_context(self):
        dataset = {
            'source': 'user_provider',
            'file_path': '/fake/path.json',
            'pmid': '123456',
            'accession': 'E-XYZ-0002'
        }
        
        with patch('os.path.exists', return_value=False):
            result = extract_dataset_metadata(dataset)
            self.assertIn("- Associated user JSON file not found or path missing.", result)
    
    def test_local_details_extractioon(self):
        dataset = {
            'source': 'user_provider',
            'organism_part': 'Liver',
            'experimental_designs': 'Case-control',
            'hardware': 'Illumina',
            'accession': 'E-MTAB-0001',
            'pmid': '123456'
        }
        
        with patch('llm_ui.app.pages.consumer_QA._fetch_pubmed_context', return_value=["Abstract: Abstract data"]), \
             patch('llm_ui.app.pages.consumer_QA._fetch_europepmc_context', return_value=[]), \
             patch('llm_ui.app.pages.consumer_QA._fetch_arrayexpress_context', return_value=[]), \
             patch('builtins.open', mock_open(read_data=json.dumps({"context": {}}))), \
             patch('os.path.exists', return_value=True):

            result = extract_dataset_metadata(dataset)
            self.assertIn("- Organism part: Liver", result)
            self.assertIn("**Fetched Publication Abstract:** Abstract data", result)

    def test_api_details_for_arrayexpress_source(self):
        dataset = {
            'source': 'arrayexpress',
            'accession': 'E-MTAB-9999',
            'pmid': '123456'
        }

        with patch('llm_ui.app.pages.consumer_QA._fetch_pubmed_context', return_value=[]), \
             patch('llm_ui.app.pages.consumer_QA._fetch_europepmc_context', return_value=[]), \
             patch('llm_ui.app.pages.consumer_QA._fetch_arrayexpress_context', return_value=["Dataset title: example", "Organism: Human"]), \
             patch('os.path.exists', return_value=False):

            result = extract_dataset_metadata(dataset)
            self.assertIn("### ArrayExpress API Specific Details", result)
            self.assertIn("Dataset title: example", result)

if __name__ == '__main__':
    unittest.main()