import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


from llm_ui.app import db_utils

DB_PATH = ":memory:"
db_utils.DB_PATH = DB_PATH

class TestDBUtils(unittest.TestCase):
    
    def setUp(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                file_path TEXT PRIMARY KEY,
                accession TEXT,
                pmid TEXT,
                title TEXT,
                organism TEXT,
                study_type TEXT,
                description TEXT,
                source TEXT,
                last_updated TIMESTAMP,
                hardware TEXT,
                organism_part TEXT,
                experimental_designs TEXT,
                assay_by_molecule TEXT,
                technology TEXT,
                sample_count TEXT,
                release_date TEXT,
                experimental_factors TEXT
            )
        ''')
    
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_accession ON datasets (accession)",
            "CREATE INDEX IF NOT EXISTS idx_pmid ON datasets (pmid)",
            "CREATE INDEX IF NOT EXISTS idx_organism ON datasets (organism)",
            "CREATE INDEX IF NOT EXISTS idx_study_type ON datasets (study_type)",
            "CREATE INDEX IF NOT EXISTS idx_source ON datasets (source)",
            "CREATE INDEX IF NOT EXISTS idx_title ON datasets (title)"
        ]
        for index_sql in indexes:
            self.cursor.execute(index_sql)

        self.conn.commit() 
        
    def tearDown(self):
        if self.conn:
          self.conn.close()
    
    def test_get_db_connection(self):
        conn = db_utils.get_db_connection()
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()
    
    def test_init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'") 
        table_exists = cursor.fetchone()
        self.assertIsNotNone(table_exists)
    
    def test_prepare_data_tuple(self):
        metadata = {
            'file_path' : 'test_path',
            'accession' : 'test_accession',
            'pmid' : '123',
            'title' : 'Test title',
            'organism' : 'Test organism',
            'study_type' : 'Test study type',
            'description' : 'Test description',
            'source' : 'Test source',
            'hardware' : 'Test hardware',
            'organism_part' : 'Test organism part',
            'experimental_designs' : 'Test experimental designs',
            'assay_by_molecule' : 'Test assay',
            'technology' : 'Test technology',
            'sample_count' : '100',
            'release_date' : '2024-01-01',
            'experimental_factors' : 'Test factors'
        }
        columns = [
            'file_path', 'accession', 'pmid', 'title', 'organism', 'study_type',
            'description', 'source', 'last_updated', 'hardware', 'organism_part',
            'experimental_designs', 'assay_by_molecule', 'technology',
            'sample_count', 'release_date', 'experimental_factors'
        ]
        result = db_utils._prepare_data_tuple(metadata, columns)
        self.assertEqual(len(result), len(columns))
        self.assertEqual(result[0], 'test_path')
        self.assertEqual(result[1], 'test_accession')
        self.assertEqual(result[2], '123')
        self.assertEqual(result[3], 'Test title')
        self.assertEqual(result[4], 'Test organism')
        self.assertEqual(result[5], 'Test study type')
        self.assertEqual(result[6], 'Test description')
        self.assertIsInstance(result[8], datetime)
        self.assertEqual(result[9], 'Test hardware')
        self.assertEqual(result[10], 'Test organism part')
        self.assertEqual(result[11], 'Test experimental designs')
        self.assertEqual(result[12], 'Test assay')
        self.assertEqual(result[13], 'Test technology')
        self.assertEqual(result[14], '100')
        self.assertEqual(result[15], '2024-01-01')
        self.assertEqual(result[16], 'Test factors')       
    
    
    def test_batch_upsert_dataset_update(self):
        initial_metadata_list=[
            {
                'file_path': 'batch_update_1',
                'accession': 'batch_accession_1',
                'pmid': '333',
                'title': 'Original Title 1',
                'organism': 'Original Organism 1',
                'study_type': 'Original Study Type 1',
                'description': 'Original Description 1',
                'source': 'Original Source 1',
                'hardware': 'Original Hardware 1',
                'organism_part': 'Original Organism Part 1',
                'experimental_designs': 'Original Experimental Designs 1',
                'assay_by_molecule': 'Original Assay 1',
                'technology': 'Original Technology 1',
                'sample_count': '700',
                'release_date': '2024-07-07',
                'experimental_factors': 'Original Factors 1'
            },
            {
                'file_path': 'batch_update_2',
                'accession': 'batch_accession_2',
                'pmid': '444',
                'title': 'Original Title 2',
                'organism': 'Original Organism 2',
                'study_type': 'Original Study Type 2',
                'description': 'Original Description 2',
                'source': 'Original Source 2',
                'hardware': 'Original Hardware 2',
                'organism_part': 'Original Organism Part 2',
                'experimental_designs': 'Original Experimental Designs 2',
                'assay_by_molecule': 'Original Assay 2',
                'technology': 'Original Technology 2',
                'sample_count': '800',
                'release_date': '2024-08-08',
                'experimental_factors': 'Original Factors 2'
            },
        ]
        db_utils.batch_upsert_datasets(initial_metadata_list)
        
        updated_metadata_list = [
            {
                'file_path': 'batch_update_1',
                'accession': 'batch_accession_1',
                'pmid': '333',
                'title': 'Updated Title A',  # Changed
                'organism': 'Updated Organism A',  # Changed
                'study_type': 'Original Study Type 1',
                'description': 'Original Description 1',
                'source': 'Original Source 1',
                'hardware': 'Original Hardware 1',
                'organism_part': 'Original Organism Part 1',
                'experimental_designs': 'Original Experimental Designs 1',
                'assay_by_molecule': 'Original Assay 1',
                'technology': 'Original Technology 1',
                'sample_count': '700',
                'release_date': '2024-07-07',
                'experimental_factors': 'Original Factors 1'
            },
            {
                'file_path': 'batch_update_2',
                'accession': 'batch_accession_2',
                'pmid': '444',
                'title': 'Updated Title B',  # Changed
                'organism': 'Updated Organism B',  # Changed
                'study_type': 'Original Study Type 2',
                'description': 'Original Description 2',
                'source': 'Original Source 2',
                'hardware': 'Original Hardware 2',
                'organism_part': 'Original Organism Part 2',
                'experimental_designs': 'Original Experimental Designs 2',
                'assay_by_molecule': 'Original Assay 2',
                'technology': 'Original Technology 2',
                'sample_count': '800',
                'release_date': '2024-08-08',
                'experimental_factors': 'Original Factors 2'
            }
        ]
        db_utils.batch_upsert_datasets(updated_metadata_list)
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM datasets WHERE file_path IN ('batch_update_1', 'batch_update_2')")
        updated_records = cursor.fetchall()
        if len(updated_records) == 2:
            record_dict = {rec['file_path']: rec for rec in updated_records}
            self.assertEqual(record_dict['batch_update_1']['title'], 'Updated Title A')
            self.assertEqual(record_dict['batch_update_1']['organism'], 'Updated Organism A')
            self.assertEqual(record_dict['batch_update_2']['title'], 'Updated Title B')
            self.assertEqual(record_dict['batch_update_2']['organism'], 'Updated Organism B')

        
    def test_fetched_pubmed_title(self):
        mock_response = MagicMock()
        mock_response.text = """
            <PubmedArticleSet>
                <PubmedArticle>
                    <MedlineCitation>
                        <Article>
                            <ArticleTitle>Test PubMed Title</ArticleTitle>
                        </Article>
                    </MedlineCitation>
                </PubmedArticle>
            </PubmedArticleSet>
        """
        mock_response.status.code = 200
        with patch('requests.get', return_value=mock_response):
            title = db_utils._fetch_pubmed_title('12345')
            self.assertEqual(title, 'Test PubMed Title')
        
        mock_response_404 = MagicMock()
        mock_response_404.text = ""
        mock_response_404.status_code = 404
        with patch('requests.get', return_value=mock_response_404):
            title = db_utils._fetch_pubmed_title('999999')
            self.assertIsNone(title)
    
    def test_extract_metadata_for_db(self):
        mock_json_data = {
            'title': 'Sample Title',
            'description': 'Sample Description',
            'organism': 'Sample Organism',
            'study type': 'Sample Study Type',
            'pmid': '12345',
            'accession': 'E-GEOD-12345',
            'section': {
                'type': 'study',
                'attributes': [
                    {'name': 'Release Date', 'value': '2024-01-15'}
                ]
            }
        }
        mock_file_path = 'mock_file.json'
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_json_data))) as mock_file:
            metadata = db_utils._extract_metadata_for_db(mock_file_path)
            expected_metadata = {
                'file_path': 'mock_file.json',
                'accession': 'E-GEOD-12345',
                'pmid': '12345',
                'title': 'Sample Title',
                'organism': 'Sample Organism',
                'study_type': 'Sample Study Type',
                'description': 'Sample Description',
                'source': 'arrayexpress',
                'hardware': None,
                'organism_part': None,
                'experimental_designs': None,
                'assay_by_molecule': None,
                'technology': None,
                'sample_count': None,
                'release_date': '2024-01-15',
                'experimental_factors': None
            }
            self.assertEqual(metadata, expected_metadata)
        
    def test_extract_metadata_from_bulk_entry(self):
        mock_entry = {
            'accession': 'bulk_accession_1',
            'title': 'Bulk Entry Title',
            'description': 'Bulk Entry Description',
            'organism': 'Bulk Organism',
            'pmid': '54321',
            'experimenttype': 'Bulk Study Type',
            'performer': 'Bulk Hardware',
            'technology': 'Bulk Technology',
            'organism_part': 'Bulk Organism Part',
            'experimental_design': 'Bulk Experimental Designs',
            'assay_name': 'Bulk Assay'
        }
        mock_file_path = 'bulk_file.json'
        result = db_utils._extract_metadata_from_bulk_entry(mock_entry, mock_file_path)
        expected_metadata = {
            'file_path': 'bulk_file.json::bulk_accession_1',
            'accession': 'bulk_accession_1',
            'pmid': '54321',
            'title': 'Bulk Entry Title',
            'organism': 'Bulk Organism',
            'study_type': 'Bulk Study Type',
            'description': 'Bulk Entry Description',
            'source': 'bulk_processed',
            'hardware': 'Bulk Hardware',
            'organism_part': 'Bulk Organism Part',
            'experimental_designs': 'Bulk Experimental Designs',
            'assay_by_molecule': 'Bulk Assay',
            'technology': 'Bulk Technology'
        }
        self.assertEqual(result, expected_metadata)
    
    def test_fetch_pubmed_titles_batch(self):
        mock_response = MagicMock()
        mock_response.text = """
            <PubmedArticleSet>
                <PubmedArticle>
                    <MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Title 1</ArticleTitle></Article></MedlineCitation>
                </PubmedArticle>
                <PubmedArticle>
                    <MedlineCitation><PMID>456</PMID><Article><ArticleTitle>Title 2</ArticleTitle></Article></MedlineCitation>
                </PubmedArticle>
            </PubmedArticleSet>
        """
        mock_response.status_code = 200
        with patch('requests.get', return_value=mock_response):
            titles = db_utils._fetch_pubmed_titles_batch(['123', '456', '789']) 
            self.assertEqual(titles, {'123': 'Title 1', '456': 'Title 2'})

    
    @patch('llm_ui.app.db_utils._extract_metadata_for_db')
    @patch('llm_ui.app.db_utils._extract_metadata_from_bulk_entry')
    @patch('llm_ui.app.db_utils.batch_upsert_datasets')
    @patch('llm_ui.app.db_utils.update_titles_from_pubmed')
    @patch('builtins.open', new_callable=mock_open, read_data='{"pmid1": {"acc": "B1"}, "pmid2": {"acc": "B2"}}')
    @patch('json.load')
    def test_scan_and_update_db(self, mock_json_load, mock_open_file, mock_update_titles, mock_batch_upsert, mock_extract_bulk, mock_extract_individual):
        mock_isdir = MagicMock(return_value=True)

        mock_glob_return = ['/mock_dir/file1.json',
                            '/mock_dir/file2.json',
                            '/mock_dir/arxpr_simplified.json']
        mock_glob = MagicMock(return_value=mock_glob_return)

        mock_bulk_data = {"pmid1": {"accession": "B1"}, "pmid2": {"accession": "B2"}}
        mock_json_load.return_value = mock_bulk_data

        with patch('os.path.isdir', mock_isdir), patch('glob.glob', mock_glob):
            mock_extract_individual.side_effect = [
                {'file_path': 'file1.json', 'title': 'Title 1'},
                {'file_path': 'file2.json', 'title': 'Title 2'}
            ]
            mock_extract_bulk.side_effect = [
                {'file_path': 'arxpr_simplified.json::B1', 'title': 'Bulk Title 1'},
                {'file_path': 'arxpr_simplified.json::B2', 'title': 'Bulk Title 2'}
            ]   

            db_utils.scan_and_update_db(['/mock_dir'])

            self.assertEqual(mock_extract_individual.call_count, 2)

            mock_open_file.assert_called_with('/mock_dir/arxpr_simplified.json', 'r', encoding='utf-8')
            mock_json_load.assert_called_once()
            
            expected_calls = [
                call([{'file_path': 'file1.json', 'title': 'Title 1'}, {'file_path': 'file2.json', 'title': 'Title 2'}]),
                call([{'file_path': 'arxpr_simplified.json::B1', 'title': 'Bulk Title 1'}, {'file_path': 'arxpr_simplified.json::B2', 'title': 'Bulk Title 2'}]),
            ]
            self.assertEqual(mock_batch_upsert.mock_calls, expected_calls)
            self.assertEqual(mock_update_titles.call_count, 1)


    @patch('llm_ui.app.db_utils.get_db_connection')
    def test_get_datasets_page(self, mock_get_conn):
        mock_get_conn.return_value = self.conn
        cursor = self.conn.cursor()
        cursor.executemany("INSERT INTO datasets (file_path, title, last_updated) VALUES (?, ?, ?)",
                   [('file_path_1', 'Title 1', '2024-01-01 10:00:00'),
                    ('file_path_2', 'Title 2', '2024-01-01 11:00:00'),
                    ('file_path_3', 'Title 3', '2024-01-01 09:00:00')])
        self.conn.commit()
        
        
        page_search = db_utils.get_datasets_page(0, 10, search_term='Title 1')
        self.assertEqual(len(page_search), 1)
        self.assertEqual(page_search[0]['title'], 'Title 1')
    
    @patch('llm_ui.app.db_utils.get_db_connection')
    def test_get_datasets_by_ids(self, mock_get_conn):
        mock_get_conn.return_value = self.conn
        cursor = self.conn.cursor()
        cursor.executemany("INSERT INTO datasets (file_path, accession, title) VALUES (?, ?, ?)",
                   [('file_path_1', 'accession_1', 'Title 1'),
                    ('file_path_2', 'accession_2', 'Title 2'),
                    ('file_path_3', 'accession_3', 'Title 3')])
        self.conn.commit()

        datasets = db_utils.get_datasets_by_ids(['accession_1', 'accession_3'])
        self.assertEqual(len(datasets), 2)
        datasets.sort(key=lambda x: x['accession'])
        self.assertEqual(datasets[0]['title'], 'Title 1')
        self.assertEqual(datasets[1]['title'], 'Title 3')

        datasets_file_path = db_utils.get_datasets_by_ids(['file_path_2'])
        self.assertEqual(len(datasets_file_path), 1)
        self.assertEqual(datasets_file_path[0]['title'], 'Title 2')

if __name__ == '__main__':
    unittest.main()

    
               