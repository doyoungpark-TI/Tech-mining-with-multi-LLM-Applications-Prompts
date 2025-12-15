# -*- coding: utf-8 -*-
"""
Technical Problem Extraction from Patent Text

This script extracts structured technical problems (Problem Targets + Symptoms)
from patent paragraphs using LLMs with Pydantic output parsing and 
LLM-as-Judge evaluation.

Author: [Author Name]
"""

# =============================================================================
# 1. Setup and Imports
# =============================================================================
import os
import json
from typing import List, Dict, Optional

import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

# =============================================================================
# 2. Pydantic Output Schema
# =============================================================================
class ProblemTarget(BaseModel):
    """A physical component or system experiencing the technical problem."""
    name: str
    code: str  # A, B, C, ...
    description: str


class ProblemSymptom(BaseModel):
    """An observable negative effect affecting the Problem Target."""
    name: str
    code: str  # 01, 02, 03, ...
    description: str
    related_targets: List[str]  # Target codes (e.g., ["A", "B"])


class TechnicalProblemOutput(BaseModel):
    """Structured output for technical problem extraction."""
    targets: List[ProblemTarget]
    symptoms: List[ProblemSymptom]


# =============================================================================
# 3. Prompt Design
# =============================================================================
POSITIVE_EXAMPLES = """
Example 1:
Input: Closing the first and second contactor device into an electrical load having a large electrical potential may cause undesired damage or reduced service life to the contactor.

Expected output:
{
  "targets": [
    {"name": "Contactor device", "code": "A", "description": "An electrical switching device used to control power flow to electrical loads with large electrical potential."}
  ],
  "symptoms": [
    {"name": "Undesired damage", "code": "01", "description": "The contactor device may suffer physical or functional damage when closing into high-potential electrical loads.", "related_targets": ["A"]},
    {"name": "Reduced service life", "code": "02", "description": "The operational lifespan of the contactor decreases due to exposure to large electrical potential during switching operations.", "related_targets": ["A"]}
  ]
}

Example 2:
Input: The current-limiting resistor has an undesired large size and weight and also radiates undesired heat in relation to the voltage applied thereacross.

Expected output:
{
  "targets": [
    {"name": "Current-limiting resistor", "code": "A", "description": "A resistor component used to limit electrical current flow in a circuit by applying voltage across it."}
  ],
  "symptoms": [
    {"name": "Large size and weight", "code": "01", "description": "The resistor has undesirably large physical dimensions and mass.", "related_targets": ["A"]},
    {"name": "Excessive heat radiation", "code": "02", "description": "The resistor generates and radiates undesired amounts of thermal energy when voltage is applied.", "related_targets": ["A"]}
  ]
}

Example 3:
Input: Specifically, it is difficult for the IPM motor to generate a torque in a high rotational speed range because a counter electromotive force is generated due to the structure of the IPM motor.

Expected output:
{
  "targets": [
    {"name": "Interior Permanent Magnet motor", "code": "A", "description": "An electric motor with permanent magnets embedded inside the rotor structure."}
  ],
  "symptoms": [
    {"name": "Difficulty generating torque at high speed", "code": "01", "description": "The motor struggles to produce adequate torque in high rotational speed ranges due to counter electromotive force.", "related_targets": ["A"]}
  ]
}
"""

def create_extraction_prompt(format_instructions: str) -> PromptTemplate:
    """Create the extraction prompt template."""
    return PromptTemplate(
        template="""[ROLE]
You are an expert AI assistant specializing in patent technical problem extraction.

[TASK]
Extract Problem Targets (components with issues) and Symptoms (negative effects) from the patent text.

[EXAMPLES]
{examples}

[INSTRUCTIONS]
1. Identify Problem Targets (physical objects with issues)
2. Identify Symptoms (negative effects on those targets)
3. Match Symptoms to Targets via codes (A/B/C for targets, 01/02/03 for symptoms)
4. Output ONLY valid JSON (no explanation, no markdown)

{format_instructions}

[INPUT]
{text}

[OUTPUT]
""",
        input_variables=["text"],
        partial_variables={
            "examples": POSITIVE_EXAMPLES,
            "format_instructions": format_instructions
        }
    )


# =============================================================================
# 4. LLM-as-Judge Evaluation Schema
# =============================================================================
class ComparisonEvaluation(BaseModel):
    """Evaluation results comparing multiple extraction candidates."""
    scores: List[float] = Field(
        description="Quality scores (0.0-1.0) for each extraction candidate"
    )
    best_candidate_index: int = Field(
        description="Index of the best extraction candidate (0-based)"
    )
    reasoning: str = Field(
        description="Comparative explanation of why the best candidate is superior"
    )


JUDGE_PROMPT_TEMPLATE = """
You are an expert patent analyst evaluating technical problem extractions.

[Evaluation Criteria - Score 0.0 to 1.0]

1. Correctness: Are Problem Targets and Symptoms accurately identified?
   - 1.0: All targets/symptoms correct, perfect separation
   - 0.5: Partially correct, some misidentification
   - 0.0: Completely incorrect

2. Completeness: Are all major problems captured with informative descriptions?
   - 1.0: All key problems, detailed descriptions
   - 0.5: Some problems missing or vague descriptions
   - 0.0: Critical omissions

3. Faithfulness: Is everything grounded in the patent text?
   - 1.0: Fully grounded, no external information
   - 0.5: Some unsupported inferences
   - 0.0: Fabricated content

[Patent Text]
{patent_text}

[Candidates]
{candidates}

[Task]
1. Identify core technical problems in patent text
2. Score each candidate on 3 criteria (average for final score)
3. Select best candidate with comparative reasoning

{format_instructions}
"""


# =============================================================================
# 5. Core Functions
# =============================================================================
def create_llm(model_name: str, api_key: str, base_url: str) -> ChatOpenAI:
    """Create LangChain ChatOpenAI instance."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0
    )


def run_extraction(chain, text: str) -> Dict:
    """Execute extraction chain and return dict result."""
    try:
        response = chain.invoke({"text": text})
        return {
            "targets": [t.dict() for t in response.targets],
            "symptoms": [s.dict() for s in response.symptoms]
        }
    except Exception as e:
        return {"error": str(e), "targets": [], "symptoms": []}


def extract_problems_batch(
    texts: List[str],
    model_name: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1"
) -> List[Dict]:
    """
    Extract technical problems from a batch of texts.
    
    Args:
        texts: List of patent paragraph texts
        model_name: LLM model identifier
        api_key: API key
        base_url: API base URL
    
    Returns:
        List of extraction results
    """
    # Initialize LLM and parser
    llm = create_llm(model_name, api_key, base_url)
    base_parser = PydanticOutputParser(pydantic_object=TechnicalProblemOutput)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    
    # Create chain
    prompt = create_extraction_prompt(base_parser.get_format_instructions())
    chain = prompt | llm | fixing_parser
    
    # Batch process
    batch_inputs = [{"text": text} for text in texts]
    results = chain.batch(batch_inputs, return_exceptions=True)
    
    # Convert results
    extractions = []
    for result in results:
        if isinstance(result, Exception):
            extractions.append({"error": str(result), "targets": [], "symptoms": []})
        else:
            extractions.append({
                "targets": [t.dict() for t in result.targets],
                "symptoms": [s.dict() for s in result.symptoms]
            })
    
    return extractions


def evaluate_extractions(
    patent_text: str,
    extractions: List[Dict],
    judge_model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1"
) -> Optional[Dict]:
    """
    Evaluate multiple extraction candidates using LLM-as-Judge.
    
    Args:
        patent_text: Original patent text
        extractions: List of extraction results to compare
        judge_model: Judge LLM model identifier
        api_key: API key
        base_url: API base URL
    
    Returns:
        Evaluation result with scores and best candidate
    """
    # Initialize
    llm = create_llm(judge_model, api_key, base_url)
    parser = JsonOutputParser(pydantic_object=ComparisonEvaluation)
    
    prompt = PromptTemplate(
        template=JUDGE_PROMPT_TEMPLATE,
        input_variables=["patent_text", "candidates"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    # Format candidates
    formatted_candidates = "\n".join([
        f"Candidate {i}:\n{json.dumps(ext, indent=2)}"
        for i, ext in enumerate(extractions)
    ])
    
    try:
        result = chain.invoke({
            "patent_text": patent_text,
            "candidates": formatted_candidates
        })
        return result
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def run_multi_model_extraction(
    df: pd.DataFrame,
    models: List[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1"
) -> pd.DataFrame:
    """
    Run extraction with multiple models and store results.
    
    Args:
        df: DataFrame with 'input' column containing patent texts
        models: List of model identifiers
        api_key: API key
        base_url: API base URL
    
    Returns:
        DataFrame with extraction results for each model
    """
    for model in models:
        model_name = model.split('/')[-1]
        print(f"Processing {model_name}...")
        
        extractions = extract_problems_batch(
            texts=df['input'].tolist(),
            model_name=model,
            api_key=api_key,
            base_url=base_url
        )
        
        df[model_name] = extractions
        print(f"  ✓ {model_name} completed")
    
    return df


# =============================================================================
# 6. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # Model groups
    LIGHTWEIGHT_MODELS = [
        'google/gemini-2.0-flash-001',
        'openai/gpt-4o-mini',
        'anthropic/claude-3.5-haiku'
    ]
    
    HP_MODELS = [
        'google/gemini-2.5-pro',
        'openai/gpt-4o',
        'anthropic/claude-3.7-sonnet'
    ]
    
    JUDGE_MODELS = [
        'google/gemini-2.5-pro',
        'openai/gpt-4o',
        'anthropic/claude-3.7-sonnet'
    ]
    
    # Example usage
    # -----------------------------------------------------------------
    # 1. Load filtered data (from problem filtering step)
    # df = pd.read_pickle("filtered_problem_paragraphs.pkl")
    
    # 2. Run multi-model extraction
    # df = run_multi_model_extraction(
    #     df=df,
    #     models=LIGHTWEIGHT_MODELS + HP_MODELS,
    #     api_key=API_KEY
    # )
    
    # 3. Evaluate with LLM-as-Judge (example for single sample)
    # sample_text = df['input'].iloc[0]
    # sample_extractions = [
    #     df['gemini-2.0-flash-001'].iloc[0],
    #     df['gpt-4o-mini'].iloc[0],
    #     df['claude-3.5-haiku'].iloc[0]
    # ]
    # 
    # eval_result = evaluate_extractions(
    #     patent_text=sample_text,
    #     extractions=sample_extractions,
    #     judge_model=JUDGE_MODELS[0],
    #     api_key=API_KEY
    # )
    # print(f"Scores: {eval_result['scores']}")
    # print(f"Best: Candidate {eval_result['best_candidate_index']}")
    
    # 4. Save results
    # df.to_pickle("problem_extractions.pkl")
    # -----------------------------------------------------------------
    
    print("Pipeline ready. Uncomment main execution block to run.")
