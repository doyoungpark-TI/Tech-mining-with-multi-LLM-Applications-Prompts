
# -*- coding: utf-8 -*-
"""
Technical Problem Filtering with Cascade LLM Ensemble

This script identifies technical problems in patent paragraphs using a cascade 
ensemble approach: lightweight LLMs (sLLM) first, then high-performance LLMs 
(HP-LLM) for uncertain cases only.

Author: [Author Name]
"""

# =============================================================================
# 1. Setup and Data Loading
# =============================================================================
import os
import re
import pickle
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load patent data
# data = pd.read_pickle("your_patent_data.pkl")

# =============================================================================
# 2. Data Preprocessing
# =============================================================================
def extract_sections(data: pd.DataFrame) -> pd.DataFrame:
    """Extract background, summary, and description sections from patent data."""
    
    for section in ['BACKGROUND', 'SUMMARY', 'DESCRIPTION']:
        col_name = section.lower()
        data[col_name] = data['description'].apply(
            lambda x: [v for k, v in x.items() if section in k]
        )
        data[col_name] = data[col_name].apply(
            lambda x: x[0] if len(x) != 0 else np.nan
        )
        # Filter paragraphs with at least 10 words
        data[col_name] = data[col_name].apply(
            lambda x: [i for i in x if len(i.split()) >= 10] if str(x) != 'nan' else x
        )
    return data


def create_paragraph_chunks(data: pd.DataFrame, max_paragraphs: int = 5) -> pd.DataFrame:
    """Create numbered paragraph chunks from patent sections."""
    
    sections = ['background', 'summary', 'description']
    rows = []
    
    for idx, row in data.iterrows():
        doc_id = row['doc_id']  # Adjust column name as needed
        
        for section in sections:
            contents = row[section]
            if str(contents) in ["[]", "nan"]:
                continue
            
            for i, text in enumerate(contents[:max_paragraphs]):
                clean_text = re.sub(r'\[\d+\]$', '', str(text))
                rows.append({
                    'doc_id': doc_id,
                    'section': section,
                    'para_idx': i,
                    'input': f"[{i+1}] {clean_text}"
                })
    
    return pd.DataFrame(rows)


# =============================================================================
# 3. Prompt Design for Technical Problem Classification
# =============================================================================
POSITIVE_EXAMPLES = """
Example 1: 
Input: "[3] Conventional lithium-ion batteries have limited energy density below 250 Wh/kg, which is insufficient to achieve the 500 km driving range required for mass-market electric vehicles."
Output: PROBLEM

Example 2:
Input: "[2] Current carbon fiber composite manufacturing requires expensive autoclave curing processes, resulting in production costs exceeding $50/kg that prohibit adoption in consumer automotive applications."
Output: PROBLEM

Example 3:
Input: "[3] Conventional photolithography processes cannot achieve feature sizes below 7 nm due to fundamental diffraction limits, preventing further miniaturization of semiconductor devices."
Output: PROBLEM
"""

PROMPT_TEMPLATE = PromptTemplate(
    template="""[ROLE]
You are an expert AI assistant specializing in patent analysis.

[TASK]
Classify whether the input paragraph describes a "Technical Problem" - a negative effect that current state-of-the-art technology cannot fully address.

[EXAMPLES]
{examples}

[INSTRUCTIONS]
1. Read the [INPUT] text carefully
2. Classify as either "PROBLEM" or "NO_PROBLEM"
3. Output ONLY the label (no explanation)

[INPUT]
{text}

[OUTPUT]
""",
    input_variables=["text"],
    partial_variables={"examples": POSITIVE_EXAMPLES}
)


# =============================================================================
# 4. LLM Configuration
# =============================================================================
def create_llm(model_name: str, api_key: str, base_url: str) -> ChatOpenAI:
    """Create LangChain ChatOpenAI instance."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0
    )


# =============================================================================
# 5. Cascade Ensemble Implementation
# =============================================================================
def run_cascade_ensemble(
    result_df: pd.DataFrame,
    sllm_models: list,
    hp_models: list,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    threshold: int = 2
) -> pd.DataFrame:
    """
    Run cascade ensemble: sLLM first, then HP-LLM for uncertain cases.
    
    Args:
        result_df: DataFrame with 'input' column
        sllm_models: List of lightweight model names
        hp_models: List of high-performance model names
        api_key: API key for model provider
        base_url: API base URL
        threshold: Majority voting threshold (default: 2 out of 3)
    
    Returns:
        DataFrame with ensemble results
    """
    batch_inputs = [{"text": text} for text in result_df["input"]]
    
    # ----- Step 1: Process with sLLM models -----
    print("=== Step 1: Processing with sLLM models ===")
    
    for model in sllm_models:
        model_name = model.split('/')[-1]
        print(f"  Processing {model_name}...")
        
        llm = create_llm(model, api_key, base_url)
        chain = PROMPT_TEMPLATE | llm | StrOutputParser()
        result_df[model_name] = chain.batch(batch_inputs, return_exceptions=True)
    
    # Convert to binary
    sllm_names = [m.split('/')[-1] for m in sllm_models]
    for name in sllm_names:
        result_df[f'{name}_binary'] = result_df[name].apply(
            lambda x: 0 if 'NO_PROBLEM' in str(x).upper() else 1
        )
    
    # sLLM ensemble (majority voting)
    sllm_votes = result_df[[f'{m}_binary' for m in sllm_names]].values
    result_df['sllm_sum'] = sllm_votes.sum(axis=1)
    result_df['sllm_ensemble'] = (result_df['sllm_sum'] >= threshold).astype(int)
    
    print(f"  sLLM ensemble distribution:\n{result_df['sllm_ensemble'].value_counts()}")
    
    # ----- Step 2: Identify uncertain cases -----
    # Uncertain = not unanimous (sum is 1 or 2 when using 3 models)
    uncertain_mask = (result_df['sllm_sum'] == 1) | (result_df['sllm_sum'] == 2)
    uncertain_indices = result_df[uncertain_mask].index.tolist()
    
    print(f"\n=== Step 2: Uncertain cases: {len(uncertain_indices)}/{len(result_df)} ===")
    
    # ----- Step 3: Process uncertain cases with HP models -----
    if len(uncertain_indices) > 0:
        print(f"\n=== Step 3: Processing uncertain cases with HP models ===")
        uncertain_inputs = [batch_inputs[i] for i in uncertain_indices]
        
        for model in hp_models:
            model_name = model.split('/')[-1]
            print(f"  Processing {model_name}...")
            
            llm = create_llm(model, api_key, base_url)
            chain = PROMPT_TEMPLATE | llm | StrOutputParser()
            hp_results = chain.batch(uncertain_inputs, return_exceptions=True)
            
            result_df[model_name] = None
            for idx, hp_result in zip(uncertain_indices, hp_results):
                result_df.loc[idx, model_name] = hp_result
        
        # HP binary conversion and ensemble
        hp_names = [m.split('/')[-1] for m in hp_models]
        for name in hp_names:
            result_df[f'{name}_binary'] = result_df[name].apply(
                lambda x: (0 if 'NO_PROBLEM' in str(x).upper() else 1) if pd.notna(x) else None
            )
        
        result_df['hp_ensemble'] = None
        for idx in uncertain_indices:
            hp_votes = [result_df.loc[idx, f'{m}_binary'] for m in hp_names]
            hp_sum = sum(v for v in hp_votes if v is not None)
            result_df.loc[idx, 'hp_ensemble'] = 1 if hp_sum >= threshold else 0
    
    # ----- Step 4: Create final cascade ensemble -----
    result_df['cascade_ensemble'] = result_df['sllm_ensemble'].copy()
    for idx in uncertain_indices:
        result_df.loc[idx, 'cascade_ensemble'] = result_df.loc[idx, 'hp_ensemble']
    
    print(f"\n=== Cascade ensemble completed ===")
    print(f"Final distribution:\n{result_df['cascade_ensemble'].value_counts()}")
    
    return result_df


# =============================================================================
# 6. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    SLLM_MODELS = [
        'google/gemini-2.0-flash-001',
        'openai/gpt-4o-mini',
        'anthropic/claude-3.5-haiku'
    ]
    
    HP_MODELS = [
        'google/gemini-2.5-pro',
        'openai/gpt-4o',
        'anthropic/claude-3.7-sonnet'
    ]
    
    # Load and preprocess data
    # data = pd.read_pickle("patent_data.pkl")
    # data = extract_sections(data)
    # result_df = create_paragraph_chunks(data)
    
    # Run cascade ensemble
    # result_df = run_cascade_ensemble(
    #     result_df=result_df,
    #     sllm_models=SLLM_MODELS,
    #     hp_models=HP_MODELS,
    #     api_key=API_KEY
    # )
    
    # Filter to PROBLEM paragraphs only
    # filtered_df = result_df[result_df['cascade_ensemble'] == 1].copy()
    # filtered_df.to_pickle("filtered_problem_paragraphs.pkl")
    
    print("Pipeline ready. Uncomment main execution block to run.")