import streamlit as st
import json

# Function to save settings to a local file
def save_settings(temp, max_token, prompt_template):
    settings = {
        'temperature': temp,
        'maximal number of tokens generated': max_token,
        'prompt_template':prompt_template
    }
    with open('settings.json', 'w') as f:
        json.dump(settings, f)
    st.success('Settings saved!')
    # Display the current settings
    st.write("Current Settings:")
    st.json(settings)

# Function to load settings from a local file
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        return {'temperature': 0.0, 'maximal number of tokens generated': ''}



def show():
    st.write("## Configure Settings")
    
    # Load existing settings
    settings = load_settings()

    # Input widgets for temp and token
    temperature = st.slider("temperature (i.e. level of creativity)", min_value = 0.01, max_value = 2.0)
    max_tokens = st.slider("Maximal number of tokens generated", min_value = 1, max_value = 500)
    prompt_template = st.text_area("Prompt template:", value = DEFAULT_PROMPT_TEMPLATE)#, height=20)

    # Save button
    if st.button("Save Settings"):
        save_settings(temperature, max_tokens, prompt_template)

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
