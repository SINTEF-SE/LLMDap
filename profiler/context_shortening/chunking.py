from llama_index.core.schema import TextNode

def _pseudo_markdown_splitter(text: str, chunk_size, chunk_overlap, markdown_headers=[], exclude_headers=[], verbose=True):
    """
    Borrowing the LangChain markdown splitter to spot header str in lieu of #, ##, ###, etc. 
    Keeps the md-formated header in medatdata dict if found, else nothing.
    Then recursively splits the text without breaking paragraphs.
    
    Args:
        text: str
            The text from UnstructuredXMLLoader in line-separated format header, subheader, pargaraphs.
        ...
        headers_to_split_on: list
            A list of tuples of the form (header, metadata_key) where header is a str that
            will be used to split the text and metadata_key is the key in the metadata dict
            that will be used to store ONLY markdown-formatted header.

    Returns:
        splits: list
            A list of Langchain doc objects, each containing paragraphs and artifact sub-headers according to 
            chunk size; and 'metadata' key would contain real markdown headers IF any.
    
    """

    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Expands and relocate this as necessary
    markdown_headers = [("METHODS", "methods"), ("METHODOLOGY", "methods"),   
                        ("RESULT", "result"), ("RESULTS", "result"),
                        ("FIG", "figure"), ("FIGURE", "figure"),
                        ("INTRO", "introduction"), ("INTRODUCTION", "introduction"),
                        ("REF", "reference"), ("REFERENCES", "reference"),
                        ("DISCUSS", "discussion"), ("DISCUSSION", "discussion"),
                        ("SUPPL", "supplement"), ("SUPPLEMENT", "supplement"),
                        ("abstract_title_1", "abstract"),
                        ]

    exclude_headers = ['reference', 'supplement']

    # split by headers
    markdown_splitter = MarkdownHeaderTextSplitter(markdown_headers)
    md_header_splits = markdown_splitter.split_text(text)

    # filter by exclusion headers
    toss = [doc for doc in md_header_splits for exclusion in exclude_headers if exclusion in doc.metadata] 
    md_filtered_splits = [doc for doc in md_header_splits if doc not in toss]    

    # chunk
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, 
            )

    splits = text_splitter.split_documents(md_filtered_splits)

    if verbose:
        print(f"md_header_splits: {len(md_header_splits)}")
        print(f"toss: {len(toss)} \n{toss}")
        print(f"md_filtered_splits: {len(md_filtered_splits)} \n{md_filtered_splits}")

    return splits


def _match_sequence(seq1: str, seq2_list: list, threshold=0.7):
    """
    Gives a sequence similarity match between seq1 and seq2 from seq2_list.

    Args: 
        seq1: str
            The sequence to match
        seq_list: list
            A list of sequences to match against seq1
        threshold: float
            The minimum similarity ratio; <.8 for short sequences seems to work well.
    Return:
        seq2: str
            The sequence that passes the matching threshold or None if insufficient match.

    """

    from difflib import SequenceMatcher

    for seq2 in seq2_list:
        if SequenceMatcher(None, seq1, seq2).ratio() > threshold:
            return seq2
    return None


def _split_non_md_headers(doc_lc, headers=[]):
    """ A secondery cleaner to extract leftover non-markdown headers from the text. 
    
    Args:
        doc_lc: LangChain Document 
            A list of Langchain doc objects, each containing paragraphs and artifact sub-headers according to 
            chunk size; and 'metadata' key would contain real markdown headers IF any.
        headers: list    
            A list of possible headers (e.g. headers = ["introduction", "paragraph", "title_1", "title_2", "fig_caption"])
    Return:
        cleaned_docs: dict
            Contains core paragraphs under dict['text'] and associated headers under dict['headers']

    """

    # Expands and relocate this as necessary
    headers = ["introduction",
                "paragraph",
                "title_1", "title_2",
                "fig_caption",
                "abstract",
                "supplementary material",
                "materials and methods",
                "results",
                "discussion",
                "results and discussion",
                "footnote_title",
                "figures and tables",
                "summary", 
                "ref",
                "lancetref",
                "experimental procedures"
                "fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9", "fig10",
                ]

    cleaned_docs = {}
    cleaned_docs['text'] = ""
    cleaned_docs['headers'] = []

    text = doc_lc.page_content

    for line in text.split("\n"):
        match = _match_sequence(line.lower(), headers)
        if match:
            cleaned_docs['headers'].append(match)
        else:
            cleaned_docs['text'] += line

    return cleaned_docs

def chunk_by_headeres_and_clean(document, chunk_size, chunk_overlap, verbose):
    """chunk a document using the headers, and remove titles and irrelevant sections
    """
    nodes = []
    splits = _pseudo_markdown_splitter(document, chunk_size, chunk_overlap, verbose=verbose)

    for doc in splits:
        split = _split_non_md_headers(doc)  # headers currently defined within the function

        if verbose:
            print(f'\nMD Headers: {doc.metadata} \nOther Headers: {split["headers"]} \nSize change: {len(doc.page_content)} ? {len(split["text"])}')
            print(f'{split["text"]}')

        # Avoid creating a node with empty text
        if len(split["text"]) == 0: continue

        nodes.append(TextNode(text=split['text']))

    return nodes
