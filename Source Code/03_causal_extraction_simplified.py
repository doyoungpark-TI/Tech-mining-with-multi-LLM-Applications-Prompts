# -*- coding: utf-8 -*-
"""
Causal Relationship Extraction from Technical Problems

This script extracts causal relationships between technical problem symptoms
(identified in previous extraction step), distinguishing between explicit
and implicit causality based on linguistic markers.

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
class CausalRelation(BaseModel):
    """A causal relationship between two symptoms."""
    source: str  # Source symptom entity
    source_code: str  # Source symptom code
    target: str  # Target symptom entity
    target_code: str  # Target symptom code
    type: str  # "explicit" or "implicit"
    connective: Optional[str] = None  # Causal marker (e.g., "caused", "resulted in")
    confidence: float = Field(ge=0.5, le=1.0)  # Confidence score


class CausalOutput(BaseModel):
    """Structured output for causal relationship extraction."""
    relations: List[CausalRelation]


# =============================================================================
# 3. Definitions and Examples
# =============================================================================
CAUSALITY_DEFINITIONS = """
[DEFINITION: Explicit Causality]
Explicit causality refers to causal relationships directly indicated through clear linguistic markers:
- Causal connectives: because, due to, since, so, therefore, as a result
- Causative verbs: cause, lead to, result in, induce, trigger, prevent
- Resultative constructions and conditional structures (if...then)
Confidence range: 0.75-1.0

[DEFINITION: Implicit Causality]
Implicit causality requires inference through context and background knowledge:
- Temporal or procedural sequences
- Physical/mechanical mechanisms
- Functional dependencies
- Ambiguous connectives requiring contextual interpretation
Confidence range: 0.5-0.75
"""

POSITIVE_EXAMPLES = """
Example 1: Explicit Causality (Direct Causal Verb)
Input:
Symptoms: [{"entity": "Valve.Leakage", "code": "V_0"}, {"entity": "System.PressureDrop", "code": "P_0"}]
Context: "The valve leakage caused a significant pressure drop in the system."

Expected output:
{
  "relations": [
    {
      "source": "Valve.Leakage",
      "source_code": "V_0",
      "target": "System.PressureDrop",
      "target_code": "P_0",
      "type": "explicit",
      "connective": "caused",
      "confidence": 0.9
    }
  ]
}

Example 2: Explicit Causality (Resultative Construction)
Input:
Symptoms: [{"entity": "Filter.Clogging", "code": "F_1"}, {"entity": "Pump.Overload", "code": "P_2"}]
Context: "Filter clogging resulted in pump overload during peak operation."

Expected output:
{
  "relations": [
    {
      "source": "Filter.Clogging",
      "source_code": "F_1",
      "target": "Pump.Overload",
      "target_code": "P_2",
      "type": "explicit",
      "connective": "resulted in",
      "confidence": 0.85
    }
  ]
}

Example 3: Implicit Causality (Conditional Dependency)
Input:
Symptoms: [{"entity": "Coolant.Depletion", "code": "C_0"}, {"entity": "Engine.Overheating", "code": "E_0"}]
Context: "When coolant levels dropped below minimum, the engine began overheating."

Expected output:
{
  "relations": [
    {
      "source": "Coolant.Depletion",
      "source_code": "C_0",
      "target": "Engine.Overheating",
      "target_code": "E_0",
      "type": "implicit",
      "connective": "when... began",
      "confidence": 0.75
    }
  ]
}
"""

NEGATIVE_GUIDELINES = """
[NEGATIVE GUIDELINES]
The following should NOT be extracted as causal relations:

1. Co-occurrence without Causation
   Example: "The system exhibited high vibration and loud noise during operation."
   Reason: Conjunction ("and") does not establish directional causality.
   Action: Return empty relations [].

2. Reverse Causality Error
   Example: "The overheating was triggered by the fan failure."
   Wrong: Overheating → Fan failure
   Correct: Fan failure → Overheating
   Reason: "X triggered/caused by Y" means Y causes X.
   Action: Extract correct direction only.

3. Inference without Textual Evidence
   Example: "Tank pressure measured at 700 bar." → ["HighPressure" → "ExplosionRisk"]
   Reason: No explicit mention of causal effect in text.
   Action: Return empty relations [].
"""


# =============================================================================
# 4. Prompt Design
# =============================================================================
def create_causal_prompt(format_instructions: str, include_negative: bool = True) -> PromptTemplate:
    """Create the causal extraction prompt template."""
    
    examples = POSITIVE_EXAMPLES
    if include_negative:
        examples += f"\n\n{NEGATIVE_GUIDELINES}"
    
    return PromptTemplate(
        template="""[ROLE]
You are an expert analyst extracting causal relationships between symptoms.

{definitions}

[EXAMPLES]
{examples}

[INSTRUCTIONS]
1. Identify causal relationships from context using provided symptom codes
2. Classify as Explicit (0.75-1.0) or Implicit (0.5-0.75) based on definitions
3. Ensure correct causal direction (source causes target)
4. Sort by confidence descending
5. Output ONLY valid JSON (no explanation, no markdown)

{format_instructions}

[INPUT]
Symptoms: {symptoms}
Context: {text_context}

[OUTPUT]
""",
        input_variables=["symptoms", "text_context"],
        partial_variables={
            "definitions": CAUSALITY_DEFINITIONS,
            "examples": examples,
            "format_instructions": format_instructions
        }
    )


# =============================================================================
# 5. LLM-as-Judge Evaluation Schema
# =============================================================================
class CausalityEvaluation(BaseModel):
    """Evaluation results comparing multiple causality extraction candidates."""
    scores: List[float] = Field(
        description="Quality scores (0.0-1.0) for each extraction candidate"
    )
    best_candidate_index: int = Field(
        description="Index of the best extraction candidate (0-based)"
    )
    reasoning: str = Field(
        description="Comparative explanation of superiority"
    )


JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluating causal relationship extractions from technical documents.

[Evaluation Criteria - Score 0.0 to 1.0]

1. Correctness: Are causal directions and types accurate?
   - 1.0: All relations correct, perfect explicit/implicit classification
   - 0.5: Some reversed directions or wrong types
   - 0.0: Completely incorrect

2. Completeness: Are all valid causal relations captured?
   - 1.0: All explicit + inferable implicit relations, appropriate confidence
   - 0.5: Missing some relations or inconsistent confidence
   - 0.0: Critical relations missing

3. Faithfulness: Is extraction grounded in text?
   - 1.0: Fully grounded, no domain assumptions without textual evidence
   - 0.5: Some unsupported inferences
   - 0.0: Fabricated relations or co-occurrence misidentified as causality

[Context Text]
{context_text}

[Available Symptoms]
{symptoms_list}

[Candidates]
{candidates}

[Task]
1. Identify valid causal relations in context
2. Score each candidate on 3 criteria (average for final score)
3. Select best candidate with comparative reasoning

{format_instructions}
"""


# =============================================================================
# 6. Core Functions
# =============================================================================
def create_llm(model_name: str, api_key: str, base_url: str) -> ChatOpenAI:
    """Create LangChain ChatOpenAI instance."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0
    )


def extract_causality_batch(
    df: pd.DataFrame,
    model_name: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    entity_col: str = "entity",
    code_col: str = "entity_code",
    context_col: str = "text"
) -> List[Dict]:
    """
    Extract causal relationships from a batch of symptom sets.
    
    Args:
        df: DataFrame with entity, entity_code, and text columns
        model_name: LLM model identifier
        api_key: API key
        base_url: API base URL
        entity_col: Column name for symptom entities
        code_col: Column name for symptom codes
        context_col: Column name for context text
    
    Returns:
        List of extraction results
    """
    # Initialize LLM and parser
    llm = create_llm(model_name, api_key, base_url)
    base_parser = PydanticOutputParser(pydantic_object=CausalOutput)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    
    # Create chain
    prompt = create_causal_prompt(base_parser.get_format_instructions())
    chain = prompt | llm | fixing_parser
    
    # Prepare batch inputs
    batch_inputs = [
        {
            "symptoms": [
                {"entity": e, "code": c}
                for e, c in zip(row[entity_col], row[code_col])
            ],
            "text_context": row[context_col]
        }
        for _, row in df.iterrows()
    ]
    
    # Batch process
    print(f"Processing {len(batch_inputs)} causality extractions...")
    results = chain.batch(batch_inputs, return_exceptions=True)
    
    # Convert results
    extractions = []
    for result in results:
        if isinstance(result, Exception):
            extractions.append({"error": str(result), "relations": []})
        else:
            extractions.append({
                "relations": [r.dict() for r in result.relations]
            })
    
    return extractions


def evaluate_causality(
    context_text: str,
    symptoms_list: List[Dict],
    extractions: List[Dict],
    judge_model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1"
) -> Optional[Dict]:
    """
    Evaluate multiple causality extraction candidates using LLM-as-Judge.
    
    Args:
        context_text: Original context text
        symptoms_list: List of symptoms with codes
        extractions: List of extraction results to compare
        judge_model: Judge LLM model identifier
        api_key: API key
        base_url: API base URL
    
    Returns:
        Evaluation result with scores and best candidate
    """
    # Initialize
    llm = create_llm(judge_model, api_key, base_url)
    parser = JsonOutputParser(pydantic_object=CausalityEvaluation)
    
    prompt = PromptTemplate(
        template=JUDGE_PROMPT_TEMPLATE,
        input_variables=["context_text", "symptoms_list", "candidates"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    # Format inputs
    formatted_candidates = "\n".join([
        f"Candidate {i}:\n{json.dumps(ext, indent=2)}"
        for i, ext in enumerate(extractions)
    ])
    
    try:
        result = chain.invoke({
            "context_text": context_text,
            "symptoms_list": json.dumps(symptoms_list, indent=2),
            "candidates": formatted_candidates
        })
        return result
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def prepare_causal_input(df: pd.DataFrame, problem_extractions: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare input data for causal extraction by creating entity-code pairs.
    
    Args:
        df: Original patent data
        problem_extractions: DataFrame with extracted problems (targets + symptoms)
    
    Returns:
        DataFrame ready for causal extraction
    """
    # Group by document and create entity lists
    df_grouped = problem_extractions.groupby('id_wisdomain').agg({
        'text': 'first',
        'target_text': list,
        'symptom_text': list,
        'target_topic_code': list,
        'symptom_topic_code': list
    }).reset_index()
    
    # Create entity and code columns
    df_grouped['entity'] = df_grouped.apply(
        lambda x: [f"{t}.{s}" for t, s in zip(x['target_text'], x['symptom_text'])],
        axis=1
    )
    df_grouped['entity_code'] = df_grouped.apply(
        lambda x: [f"{tc}{sc}" for tc, sc in zip(x['target_topic_code'], x['symptom_topic_code'])],
        axis=1
    )
    
    return df_grouped


# =============================================================================
# 7. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # Model selection
    EXTRACTION_MODEL = 'openai/gpt-4o-mini'  # Cost-effective for extraction
    
    JUDGE_MODELS = [
        'google/gemini-2.5-pro',
        'openai/gpt-4o',
        'anthropic/claude-3.7-sonnet'
    ]
    
    # Example usage
    # -----------------------------------------------------------------
    # 1. Load data with extracted problems
    # df = pd.read_pickle("problem_extractions.pkl")
    
    # 2. Prepare causal input (entity-code pairs)
    # df_causal = prepare_causal_input(df, problem_extractions)
    
    # 3. Extract causal relationships
    # extractions = extract_causality_batch(
    #     df=df_causal,
    #     model_name=EXTRACTION_MODEL,
    #     api_key=API_KEY
    # )
    # df_causal['causality'] = extractions
    
    # 4. Evaluate with LLM-as-Judge (example for single sample)
    # sample_idx = 0
    # symptoms = [
    #     {"entity": e, "code": c}
    #     for e, c in zip(df_causal['entity'].iloc[sample_idx],
    #                     df_causal['entity_code'].iloc[sample_idx])
    # ]
    # eval_result = evaluate_causality(
    #     context_text=df_causal['text'].iloc[sample_idx],
    #     symptoms_list=symptoms,
    #     extractions=[extractions[sample_idx]],
    #     judge_model=JUDGE_MODELS[0],
    #     api_key=API_KEY
    # )
    
    # 5. Save results
    # df_causal.to_pickle("causal_extractions.pkl")
    # -----------------------------------------------------------------
    
    print("Pipeline ready. Uncomment main execution block to run.")
