import streamlit as st
import json
import os

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """You are an AI language model assistant. Your task is to answer any question the user may have, by using information in the title, abstract or the full text of biomedical paper if this is provided.

EXAMPLE TASK:
Title:
Multi-omics profiling of younger Asian breast cancers reveals distinctive molecular signatures.

Abstract:
Breast cancer (BC) in the Asia Pacific regions is enriched in younger patients and rapidly rising in incidence yet its molecular bases remain poorly characterized. Here we analyze the whole exomes and transcriptomes of 187 primary tumors from a Korean BC cohort (SMC) enriched in pre-menopausal patients and perform systematic comparison with a primarily Caucasian and post-menopausal BC cohort (TCGA). SMC harbors higher proportions of HER2+ and Luminal B subtypes, lower proportion of Luminal A with decreased ESR1 expression compared to TCGA. We also observe increased mutation prevalence affecting BRCA1, BRCA2, and TP53 in SMC with an enrichment of a mutation signature linked to homologous recombination repair deficiency in TNBC. Finally, virtual microdissection and multivariate analyses reveal that Korean BC status is independently associated with increased TIL and decreased TGF-β signaling expression signatures, suggesting that younger Asian BCs harbor more immune-active microenvironment than western BCs.

Question:
What is the cancer type discussed in the paper?

Your answer: Breast cancer
(END OF EXAMPLE)

ACTUAL TASK:
Title:
{title}

Abstract:
{abstract}

Question:
{question}

Your answer: """

# Function to save settings to a local file
def save_settings(settings_dict):
    with open('settings.json', 'w') as f:
        json.dump(settings_dict, f)
    st.success('Settings saved! These will be used in all future interactions.')
    return settings_dict

# Function to load settings from a local file
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        # Default settings
        default_settings = {
            'temperature': 0.3,
            'max_tokens': 500,
            'prompt_template': DEFAULT_PROMPT_TEMPLATE,
            'model': 'llama3.1I-8b-q4',
            'use_openai': False,
            'similarity_k': 5,
            'profiler_options': {
                'field_info_to_compare': 'choices'
            }
        }
        # Create the settings file with defaults
        save_settings(default_settings)
        return default_settings

def show():
    st.title("Configuration Settings")
    st.write("Adjust application settings to customize LLM behavior and profiler options.")
    
    # Load existing settings
    settings = load_settings()
    
    # Create tabs for different settings categories
    llm_tab, profiler_tab, advanced_tab = st.tabs(["LLM Settings", "Profiler Settings", "Advanced Options"])
    
    with llm_tab:
        st.subheader("Language Model Settings")
        
        # Model selection
        st.write("#### Model Selection")
        use_openai = st.toggle("Use OpenAI API", value=settings.get('use_openai', False),
                              help="Toggle to use OpenAI's API or local models")
        
        if use_openai:
            model = st.selectbox("OpenAI Model", 
                                ["4o", "4om"], 
                                index=0 if settings.get('model') == "4o" else 1,
                                format_func=lambda x: "GPT-4o" if x == "4o" else "GPT-4o-mini",
                                help="Select which OpenAI model to use")
            
            # Check if OpenAI API key is set
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("OpenAI API Key is not set! Please set the OPENAI_API_KEY environment variable.")
        else:
            model = st.selectbox("Local Model", 
                                ["llama3.1I-8b-q4", "biolm", "ministral_gguf", "ds8b-i4"], 
                                index=0,
                                format_func=lambda x: {
                                    "llama3.1I-8b-q4": "Llama 3.1 (8B, INT4)",
                                    "biolm": "Llama3-OpenBioLLM-8B",
                                    "ministral_gguf": "Ministral-8B-Instruct",
                                    "ds8b-i4": "DeepSeek-R1-Distill-Llama-8B"
                                }.get(x, x),
                                help="Select which local model to use")
        
        # Response parameters
        st.write("#### Response Parameters")
        temperature = st.slider("Temperature", 
                               min_value=0.0, 
                               max_value=2.0, 
                               value=float(settings.get('temperature', 0.3)),
                               step=0.1,
                               help="Higher values make output more random, lower values make it more deterministic")
        
        max_tokens = st.slider("Maximum Output Tokens", 
                              min_value=50, 
                              max_value=2000, 
                              value=int(settings.get('max_tokens', 500)),
                              step=50,
                              help="Maximum number of tokens to generate in the response")
    
    with profiler_tab:
        st.subheader("Paper Profiler Settings")
        
        similarity_k = st.slider("Similarity K", 
                                min_value=1, 
                                max_value=10, 
                                value=int(settings.get('similarity_k', 5)),
                                help="Number of similar chunks to retrieve")
        
        field_info = st.selectbox("Field Info to Compare", 
                                 ["choices", "direct"],
                                 index=0 if settings.get('profiler_options', {}).get('field_info_to_compare') == "choices" else 1,
                                 help="Method to compare fields")
    
    with advanced_tab:
        st.subheader("Advanced Settings")
        
        st.write("#### Prompt Template")
        st.write("Use placeholders like `{title}`, `{abstract}`, `{question}`, and `{datasets}` in your template.")
        prompt_template = st.text_area("Custom Prompt Template", 
                                      value=settings.get('prompt_template', DEFAULT_PROMPT_TEMPLATE),
                                      height=300,
                                      help="Template used for generating prompts when interacting with papers")
        
        # Reset to default button
        if st.button("Reset to Default Template"):
            prompt_template = DEFAULT_PROMPT_TEMPLATE
            st.info("Template reset to default. Click 'Save Settings' to apply.")
    
    # Save button
    if st.button("Save Settings"):
        updated_settings = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'prompt_template': prompt_template,
            'model': model,
            'use_openai': use_openai,
            'similarity_k': similarity_k,
            'profiler_options': {
                'field_info_to_compare': field_info
            }
        }
        settings = save_settings(updated_settings)
        
    # Display current settings
    with st.expander("Current Configuration", expanded=False):
        st.json(settings)
