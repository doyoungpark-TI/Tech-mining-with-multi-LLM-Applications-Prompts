# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:19:42 2025

@author: tmlab
"""


import pickle
import pandas as pd
import numpy as np

# Specify the path to your .pkl file
file_path = 'D:/OneDrive/연구/04_TPA_tfsc_draft/data/data_problems_symptoms.pkl'

import pandas as pd

# pickle 파일을 읽어서 데이터프레임으로 불러오기
df = pd.read_pickle("./chain_final.pkl")

file_path = r'D:\OneDrive\연구\04_TPA_tfsc_draft\251218_리비전\연구결과.xlsx'
result_topics_target = pd.read_excel(file_path, sheet_name = 'target')
result_topics_symptom = pd.read_excel(file_path, sheet_name = 'symptom')

# load claims_decomposed
from huggingface_hub import hf_hub_download
import pickle
import pandas as pd

# 다운드할 파일과 리포지토리 설정
repo_id = "sanghyyyyyyyun/patent"

# load
upload_path = "Y02E60_32_F17C5_prep_v2.pkl"  # Hugging Face Hub에 저장될 경로

# 파일 다운로드
local_file = hf_hub_download(
    repo_id=repo_id,
    filename=upload_path,
    repo_type="dataset"
)

print(f"파일 다운로드 완료: {local_file}")

with open(local_file, "rb") as f:
    data_meta = pickle.load(f)
    

#%% 01 prep data
    
symptom_dict = dict(zip(result_topics_symptom['Topic'], result_topics_symptom['Label']))
target_dict = dict(zip(result_topics_target['Topic'], result_topics_target['Label']))

# preprocessing

id2year_dict = dict(zip(data_meta['번호'], data_meta['year_application']))
df['year_application'] = df['id_wisdomain'].apply(lambda x : id2year_dict[x])

id2title_dict = dict(zip(data_meta['번호'], data_meta['명칭(원문)']))
df['title'] = df['id_wisdomain'].apply(lambda x : id2title_dict[x])

id2abstract_dict = dict(zip(data_meta['번호'], data_meta['요약(원문)']))
df['abstract'] = df['id_wisdomain'].apply(lambda x : id2abstract_dict[x])

id2claim1st_dict = dict(zip(data_meta['번호'], data_meta['대표 청구항']))
df['claim1st'] = df['id_wisdomain'].apply(lambda x : id2claim1st_dict[x])

related_topic = [8, 13, 22, 10, 28, 27]

# for topic in related_topic : 
    
df['symptom_topic_filtered'] = df['symptom_topic'].apply(lambda x : list(set([i for i in x if i in related_topic])))
# Input
# 연관 문제 code : 문제명
df['symptom_topic_filtered_text'] = ""

for idx, row in df.iterrows() : 
    code_list = row['symptom_topic_filtered']
    text = ""
    for code in code_list : 
        text += "{} : ".format(code)
        text +=  symptom_dict[code]
        text +=  "\n"
    
    df['symptom_topic_filtered_text'][idx] = text
    
# solution data filtering
data_solution = df.loc[df['symptom_topic_filtered'].apply(lambda x : True if len(x) != 0 else False) , :].reset_index(drop = 1)
# data_sample = data_solution.sample(int(len(data_solution)*0.1),  random_state = 1234).reset_index(drop = 1)



#%% 03. llm loading

from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
import os

# ==========================================
# 1. 모델 선택 (사용할 모델의 주석을 해제하세요)
# ==========================================

model_name = "anthropic/claude-3.7-sonnet"               # (현재 실제 작동 모델 예시)

# ==========================================
# 2. OpenRouter 클라이언트 통합 초기화
# ==========================================
llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    max_retries=3,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"), # OpenRouter API Key
    openai_api_base="https://openrouter.ai/api/v1", # OpenRouter Base URL
    # OpenRouter 랭킹/통계를 위한 선택적 헤더 (필요 시 추가)
    # default_headers={
    #     "HTTP-Referer": "https://your-site.com", 
    #     "X-Title": "Your App Name"
    # }
)

# 테스트 출력
print(f"Current Model: {llm.model_name}")


#%% 04. prompt loading
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# 1. 외부 변수 정의 (섹션별 텍스트 컨텐츠)
# ==============================================================================

definition_text = """
[DEFINITION: Technical Solution]
A "Technical Solution" is the technical means that an invention proposes to address one or more technical problems - specifically, a method, apparatus, system, or composition that eliminates, mitigates, or overcomes the identified negative effects.

[DEFINITION: Solution Types]
- Method: A process, procedure, or sequence of steps to solve the problem (HOW the solution operates)
- Apparatus: A physical device, component, or tangible structure that solves the problem (WHAT physical entity solves it)
- System: An integrated architecture combining multiple components that work together to solve the problem (HOW components coordinate)
"""

positive_examples_data = """
Example 1:
Input:
Problems: 0: Nut loosening | 1: Assembly complexity
Patent: "The fastener includes a nylon insert ring embedded within the internal threads. This insert creates frictional interference with the mating screw, eliminating the need for lock washers."

Reasoning:
1. Solution: "Nylon insert ring embedded in internal threads"
2. Type: Physical component → "Apparatus"
3. Problems: Frictional interference prevents loosening (0), eliminates lock washers (1)
4. Description: Nylon deforms against threads, creating friction to prevent rotation without extra parts.

Output:
{
  "solutions": [
    {
      "solution": "Nylon insert ring embedded in internal threads",
      "solution_type": "Apparatus",
      "related_problem_ids": [0, 1],
      "description": "The nylon ring creates frictional interference with mating threads, preventing loosening under vibration and eliminating the need for lock washers."
    }
  ]
}

Example 2:
Input:
Problems: 5: High power consumption | 8: Screen overheating
Patent: "The display controller implements a dynamic dimming module that adjusts the backlight duty cycle based on the average picture level (APL). By reducing current during dark scenes, thermal generation is minimized."

Reasoning:
1. Solution: "Dynamic dimming module based on APL"
2. Type: Integrated controller + sensor + logic → "System"
3. Problems: Reduced current lowers power (5) and heat (8)
4. Description: Analyzes video brightness to adjust backlight power dynamically.

Output:
{
  "solutions": [
    {
      "solution": "Dynamic dimming module based on Average Picture Level",
      "solution_type": "System",
      "related_problem_ids": [5, 8],
      "description": "The module analyzes video brightness and reduces backlight duty cycle during dark scenes, directly decreasing power consumption and thermal generation."
    }
  ]
}

Example 3:
Input:
Problems: 0: Prolonged refueling time | 13: Temperature overshoot
Patent: "The system uses a predictive algorithm to terminate refueling at an upper state of charge based on pre-estimated correction values, compensating for pressure drop after valve closure."

Reasoning:
1. Solution: "Terminate refueling at pre-calculated upper charge level"
2. Type: Algorithmic procedure → "Method"
3. Problems: Single-cycle filling reduces time (0), compensation prevents overshoot (13)
4. Description: Fills above 100% using correction values for post-closure effects.

Output:
{
  "solutions": [
    {
      "solution": "Terminate refueling at pre-calculated upper charge level",
      "solution_type": "Method",
      "related_problem_ids": [0, 13],
      "description": "The algorithm fills above rated density using pre-estimated correction values for pressure drop and thermal stabilization, achieving target charge in one cycle without temperature violations."
    }
  ]
}
"""

negative_guidelines = """
[NEGATIVE GUIDELINES]
The following should NOT be extracted as technical solutions:

1. Mere results/benefits (without mechanism)
   Example: "The invention reduces manufacturing costs by 20%."
   → Describes outcome/effect, not the technical means to achieve it. Do NOT extract.
   → Instead extract: "Use molded unitary body structure" (the mechanism)

2. Problem restatement (goal without means)
   Example: "The system addresses the issue of leakage in high-pressure environments."
   → Restates problem/goal without specifying technical means. Do NOT extract.
   → Instead extract: "Employ double O-ring seal configuration" (the specific means)

3. Overly generic/abstract solutions
   Example: "Optimize the control algorithm."
   → Too vague without specific technical content. Do NOT extract.
   → Instead extract: "Implement PID feedback control with adaptive gain tuning"
"""


    
#%% 05. Solution Extraction using LLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.01, max_retries=3)

# ===== 1) Pydantic 모델 =====
class Solution(BaseModel):
    solution: str = Field(description="Clear solution statement (5-20 words)")
    solution_type: Literal["Method", "Apparatus", "System"]
    related_problem_ids: List[int] = Field(description="Problem IDs addressed")
    description: str = Field(description="Technical explanation (2-4 sentences)")

class SolutionOutput(BaseModel):
    solutions: List[Solution]

# ===== 2) 파서 구성 =====
base_parser = PydanticOutputParser(pydantic_object=SolutionOutput)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
format_instructions = base_parser.get_format_instructions()

# ===== 3) 동적 프롬프트 생성 =====
def create_solution_prompt(include_definition=True, include_negative=True):
    def_block = definition_text if include_definition else ""
    
    examples = positive_examples_data
    if include_negative:
        examples += f"\n{negative_guidelines}"
    
    return PromptTemplate(
        template="""
[ROLE]
You are an expert patent analyst extracting technical solutions from patent documents.

{definition_block}

[TASK]
Extract 2-5 core solutions, classify by type, map to problem IDs.

[RULES]
- Solution: Clear statement (5-20 words), focus on WHAT not HOW
- Classification: Steps → Method | Physical thing → Apparatus | Multiple components → System
- Mapping: Only to explicitly addressed problems
- Description: HOW solution addresses problems (2-4 sentences)

[NEGATIVE GUIDELINES]
Do NOT extract:
1. Mere results/benefits without mechanism
2. Prior art description
3. Vague marketing claims
4. Problem restatement without means

{examples_block}

[INPUT]
Problems: {problems}
Patent: {patent_content}

{format_instructions}
""",
        input_variables=["problems", "patent_content"],
        partial_variables={
            "definition_block": def_block,
            "examples_block": examples,
            "format_instructions": format_instructions
        }
    )

# ===== 4) 4가지 버전 생성 =====
prompt_v1 = create_solution_prompt(include_definition=False, include_negative=False)
prompt_v2 = create_solution_prompt(include_definition=True, include_negative=False)
prompt_v3 = create_solution_prompt(include_definition=False, include_negative=True)
prompt_v4 = create_solution_prompt(include_definition=True, include_negative=True)

chain_v1 = prompt_v1 | llm | fixing_parser
chain_v2 = prompt_v2 | llm | fixing_parser
chain_v3 = prompt_v3 | llm | fixing_parser
chain_v4 = prompt_v4 | llm | fixing_parser

#%% ===== 5) 실행 =====
idx = 1

# 입력 데이터 준비
problems = df.loc[idx, 'symptom_topic_filtered_text']
patent_content = df.loc[idx, 'abstract'] + "\n\n" + df.loc[idx, 'claim1st']

# 4가지 버전 실행
response_v1 = chain_v1.invoke({"problems": problems, "patent_content": patent_content})
response_v2 = chain_v2.invoke({"problems": problems, "patent_content": patent_content})
response_v3 = chain_v3.invoke({"problems": problems, "patent_content": patent_content})
response_v4 = chain_v4.invoke({"problems": problems, "patent_content": patent_content})

# ===== 6) 결과 변환 =====
result_v1 = {"solutions": [s.dict() for s in response_v1.solutions]}
result_v2 = {"solutions": [s.dict() for s in response_v2.solutions]}
result_v3 = {"solutions": [s.dict() for s in response_v3.solutions]}
result_v4 = {"solutions": [s.dict() for s in response_v4.solutions]}

# ===== 7) 출력 =====
import json
print("=== V1: Baseline (No Def, No Neg) ===")
print(json.dumps(result_v1, indent=2, ensure_ascii=False))
print("\n=== V2: With Definition ===")
print(json.dumps(result_v2, indent=2, ensure_ascii=False))
print("\n=== V3: With Negative Guidelines ===")
print(json.dumps(result_v3, indent=2, ensure_ascii=False))
print("\n=== V4: Full (Def + Neg) ===")
print(json.dumps(result_v4, indent=2, ensure_ascii=False))

#%% 06. batch output


data_sample = data_solution.sample(int(len(data_solution)*0.1),  random_state = 1234).reset_index(drop = 1)


batch_inputs = [
    {
        "problems": row['symptom_topic_filtered_text'],
        "patent_content": row['abstract'] + "\n\n" + row['claim1st']
    }
    for _, row in data_sample.iterrows()
]

for version_name, chain in [
    ('v1_baseline', chain_v1), 
    ('v2_with_def', chain_v2), 
    ('v3_with_neg', chain_v3), 
    ('v4_full', chain_v4)
]:
    print(f"Processing {version_name}...")
    results = chain.batch(batch_inputs, return_exceptions= True)
    data_sample[f'solution_{version_name}'] = [
        {"solutions": [s.dict() for s in resp.solutions]} 
        for resp in results
    ]

# ===== 결과 확인 =====
import json

idx = 0
print(f"\n=== Patent {idx} Results ===")
print(f"Problems: {data_sample.iloc[idx]['symptom_topic_filtered_text']}\n")

for version in ['v1_baseline', 'v2_with_def', 'v3_with_neg', 'v4_full']:
    print(f"\n--- {version} ---")
    print(json.dumps(data_sample.iloc[idx][f'solution_{version}'], 
                     indent=2, ensure_ascii=False))


#%% 07. Solution Judge 평가

import json
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# --- Output Schema ---
class SolutionEvaluation(BaseModel):
    """Evaluation results comparing multiple solution extraction candidates."""
    scores: List[float] = Field(
        description="Quality scores (0.0-1.0) for each extraction candidate"
    )
    best_candidate_index: int = Field(
        description="Index of the best extraction candidate (0-based)"
    )
    reasoning: str = Field(
        description="Detailed comparative explanation of superiority"
    )

# --- Evaluation Prompt ---
solution_judge_template = """
You are an expert evaluating technical solution extractions from patent documents.

[Evaluation Criteria - Score 0.0 to 1.0]

1. Correctness: Are solution types and problem mappings accurate?
   - 1.0: Perfect type classification (Method/Apparatus/System), accurate problem mapping
   - 0.75: Mostly correct types, minor mapping issues
   - 0.5: Some type confusion or incorrect mappings
   - 0.25: Major type misclassification
   - 0.0: Completely incorrect types or mappings

2. Completeness: Are all core solutions captured?
   - 1.0: All core solutions (2-5) extracted, no redundancy
   - 0.75: Most solutions captured, reasonable coverage
   - 0.5: Missing key solutions or excessive redundancy
   - 0.25: Major omissions or irrelevant extractions
   - 0.0: Critical solutions missing

3. Faithfulness: Is extraction grounded in patent text?
   - 1.0: All solutions with clear mechanisms from text, no results/benefits confused as solutions
   - 0.75: Mostly grounded, minor interpretation
   - 0.5: Some prior art or vague claims extracted
   - 0.25: Extracted results instead of mechanisms
   - 0.0: Fabricated solutions or problem restatements

[Problems]
{problems}

[Patent Content]
{patent_content}

[Candidates]
{candidates}

[Task]
1. Identify valid technical solutions in patent
2. Score each candidate on 3 criteria (average for final score)
3. Select best candidate with comparative reasoning

{format_instructions}
"""

# --- Evaluation Function ---
def compare_solution_candidates(
    problems: str,
    patent_content: str,
    extractions: List[Dict],
    model = None
) -> Optional[dict]:
    """
    Compare multiple LLM solution extraction results without ground truth.
    
    Args:
        problems: List of identified technical problems
        patent_content: Patent abstract and claims
        extractions: List of solution extraction results from different LLMs
        model: Judge LLM model
    """
    parser = JsonOutputParser(pydantic_object=SolutionEvaluation)
    
    prompt = PromptTemplate(
        template=solution_judge_template,
        input_variables=["problems", "patent_content", "candidates"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | model | parser
    
    formatted_candidates = "\n".join([
        f"Candidate {i}:\n{json.dumps(ext, indent=2, ensure_ascii=False)}" 
        for i, ext in enumerate(extractions)
    ])
    
    try:
        result = chain.invoke({
            "problems": problems,
            "patent_content": patent_content,
            "candidates": formatted_candidates
        })
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

#%% 08. Batch Solution Judge

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import json

# Configuration
judge_models = ['google/gemini-2.5-pro', 'openai/gpt-5', 'x-ai/grok-4']

# ✅ 올바른 컬럼명 사용 (실제 df에 저장된 이름)
prompt_versions = ['solution_v1_baseline', 'solution_v2_with_def', 
                   'solution_v3_with_neg', 'solution_v4_full']

permutations = [[0,1,2,3], [2,3,0,1]]


# Parser & Prompt
parser = JsonOutputParser(pydantic_object=SolutionEvaluation)
prompt = PromptTemplate(
    template=solution_judge_template,
    input_variables=["problems", "patent_content", "candidates"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Main Loop
for judge_model in judge_models:
    judge_name = judge_model.split('/')[-1]
    print(f"\n=== Processing {judge_name} ===")
    
    llm = ChatOpenAI(
        model=judge_model,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.00,
        max_retries=3
    )
    
    chain = prompt | llm | parser
    
    # Prepare batch inputs
    batch_inputs = []
    row_perm_map = []
    
    for idx, row in data_sample.iterrows():
        problems = row['symptom_topic_filtered_text']
        patent_content = row['abstract'] + "\n\n" + row['claim1st']
        
        # ✅ 각 row에 모든 버전이 있는지 확인
        if not all(col in row.index and row[col] is not None for col in prompt_versions):
            print(f"Skipping row {idx}: missing solution columns")
            continue
        
        for perm_idx, perm in enumerate(permutations):
            extractions = [row[prompt_versions[i]] for i in perm]
            
            # Skip if any extraction is empty or invalid
            if any(not isinstance(ext, dict) or not ext.get('solutions') for ext in extractions):
                continue
            
            formatted_candidates = "\n".join([
                f"Candidate {i}:\n{json.dumps(ext, indent=2, ensure_ascii=False)}" 
                for i, ext in enumerate(extractions)
            ])
            
            batch_inputs.append({
                "problems": problems,
                "patent_content": patent_content,
                "candidates": formatted_candidates
            })
            row_perm_map.append((idx, perm_idx, perm))
    
    # Batch execution
    print(f"Evaluating {len(batch_inputs)} cases...")
    
    if len(batch_inputs) == 0:
        print("No valid inputs to process!")
        continue
    
    results = chain.batch(
        batch_inputs,
        config={"max_concurrency": 50},
        return_exceptions=True
    )
    
    # Store results
    data_sample[f'solution_judge_{judge_name}'] = [[] for _ in range(len(data_sample))]
    
    error_count = 0
    for (row_idx, perm_idx, perm), result in zip(row_perm_map, results):
        if isinstance(result, Exception):
            error_count += 1
            print(f"  Error at row {row_idx}, perm {perm_idx}: {str(result)[:100]}")
            continue
        
        data_sample.at[row_idx, f'solution_judge_{judge_name}'].append({
            'perm': perm,
            'scores': result['scores'],
            'best_idx': result['best_candidate_index'],
            'reasoning': result['reasoning']
        })
    
    print(f"✓ Complete ({error_count} errors)")

print("\n✓ All solution judges complete")

data_sample.to_pickle("./data_solution_eval.pkl")


#%% 09. 결과 분석 및 시각화
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 데이터 로드
data_sample = pd.read_pickle("./data_solution_eval.pkl")

# 설정
prompt_versions = ['v1_baseline', 'v2_with_def', 'v3_with_neg', 'v4_full']
judge_cols = ['solution_judge_gemini-2.5-pro', 'solution_judge_gpt-5', 'solution_judge_grok-4']
permutations = [[0,1,2,3], [2,3,0,1]]

# 프롬프트별 점수 계산
data_sample['prompt_scores'] = [{} for _ in range(len(data_sample))]

for idx, row in data_sample.iterrows():
    print(f"{idx}/{len(data_sample)}", end='\r')
    
    for judge_col in judge_cols:
        results = row[judge_col]
        if not results:
            continue
        
        scores = [0, 0, 0, 0]
        count = 0
        
        for result in results:
            if result is None:
                continue
            perm = result['perm']
            for i, score in enumerate(result['scores']):
                scores[perm[i]] += score
            count += 1
        
        if count == 0:
            continue
        
        avg_scores = [s/count for s in scores]
        
        for i, pv in enumerate(prompt_versions):
            if pv not in data_sample.at[idx, 'prompt_scores']:
                data_sample.at[idx, 'prompt_scores'][pv] = []
            data_sample.at[idx, 'prompt_scores'][pv].append(avg_scores[i])

# 최종 점수 (3 judges 평균)
for pv in prompt_versions:
    data_sample[f'final_{pv}'] = data_sample['prompt_scores'].apply(
        lambda x: np.mean(x[pv]) if pv in x and x[pv] else np.nan
    )

# 시각화
final_cols = [f'final_{pv}' for pv in prompt_versions]
df_viz = data_sample[final_cols].reset_index(drop=True)
df_viz = df_viz.rename(columns={
    'final_v1_baseline': 'V1: Baseline',
    'final_v2_with_def': 'V2: + Definition',
    'final_v3_with_neg': 'V3: + Negative',
    'final_v4_full': 'V4: Full'
})

long = df_viz.melt(var_name='prompt', value_name='score').dropna()

palette = {
    'V1: Baseline': '#4C78A8', 
    'V2: + Definition': '#F58518',
    'V3: + Negative': '#54A24B', 
    'V4: Full': '#E45756'
}

plt.figure(figsize=(10, 5))
ax = sns.violinplot(data=long, x='prompt', y='score', palette=palette, inner=None)
sns.stripplot(data=long, x='prompt', y='score', size=3, alpha=0.4, ax=ax, color='black')

# Median 표시
medians = long.groupby('prompt')['score'].median()
for i, (p, md) in enumerate(medians.items()):
    ax.hlines(md, i-0.35, i+0.35, linestyles='--', color='red', linewidth=2)
    ax.text(i, md+0.01, f'{md:.3f}', ha='center', va='bottom', 
            fontsize=12, fontweight='bold')

ax.set_title('Solution Extraction Performance by Prompt Version', fontsize=14, fontweight='bold')
ax.set_ylabel('Judge Score', fontsize=12)
ax.set_xlabel('')
ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.show()

print("\n=== Summary Statistics ===")
print(df_viz.describe())

# ===== Judge별 상세 분석 =====
judge_comparison = []

for judge_col in judge_cols:
    judge_name = judge_col.replace('solution_judge_', '')
    
    for pv in prompt_versions:
        scores = []
        for idx, row in data_sample.iterrows():
            if pv in row['prompt_scores'] and row['prompt_scores'][pv]:
                # 해당 judge의 점수만 추출 (judge_cols 순서에 맞춰)
                judge_idx = judge_cols.index(judge_col)
                if len(row['prompt_scores'][pv]) > judge_idx:
                    scores.append(row['prompt_scores'][pv][judge_idx])
        
        if scores:
            judge_comparison.append({
                'Judge': judge_name,
                'Version': pv,
                'Mean': np.mean(scores),
                'Median': np.median(scores),
                'Std': np.std(scores),
                'Count': len(scores)
            })

df_judge = pd.DataFrame(judge_comparison)
print("\n=== Judge-wise Comparison ===")
print(df_judge.pivot_table(index='Version', columns='Judge', values='Mean'))

# ===== Best Version 선택 빈도 =====
best_version_counts = {pv: 0 for pv in prompt_versions}

for idx, row in data_sample.iterrows():
    final_scores = [row[f'final_{pv}'] for pv in prompt_versions]
    
    if not any(np.isnan(final_scores)):
        best_idx = np.argmax(final_scores)
        best_version_counts[prompt_versions[best_idx]] += 1

print("\n=== Best Version Frequency ===")
for pv, count in best_version_counts.items():
    print(f"{pv}: {count} ({count/len(data_sample)*100:.1f}%)")

# ===== Pairwise Comparison =====
from scipy import stats

print("\n=== Pairwise t-test (Bonferroni corrected) ===")
alpha = 0.05
n_comparisons = 6  # 4C2 = 6

for i in range(len(prompt_versions)):
    for j in range(i+1, len(prompt_versions)):
        v1 = prompt_versions[i]
        v2 = prompt_versions[j]
        
        scores1 = data_sample[f'final_{v1}'].dropna()
        scores2 = data_sample[f'final_{v2}'].dropna()
        
        if len(scores1) > 0 and len(scores2) > 0:
            t_stat, p_val = stats.ttest_rel(scores1, scores2)
            corrected_p = min(p_val * n_comparisons, 1.0)
            sig = "***" if corrected_p < 0.001 else "**" if corrected_p < 0.01 else "*" if corrected_p < 0.05 else "n.s."
            
            print(f"{v1} vs {v2}: t={t_stat:.3f}, p={corrected_p:.4f} {sig}")
            
# ===== 기술통계 요약 =====
print("\n=== Summary Statistics ===")
summary_stats = []
for pv in prompt_versions:
    scores = data_sample[f'final_{pv}'].dropna()
    if len(scores) > 0:
        summary_stats.append({
            'Version': pv,
            'Mean': np.mean(scores),
            'Median': np.median(scores),
            'Std': np.std(scores),
            'Min': np.min(scores),
            'Max': np.max(scores),
            'Count': len(scores)
        })

df_summary = pd.DataFrame(summary_stats)
print(df_summary.to_string(index=False))
    

# ===== 저장 =====
data_sample.to_pickle("./data_solution_analyzed.pkl")
df_viz.to_csv("./solution_scores.csv", index=False)
print("\n✓ Results saved!")


#%% 10. thffntus
import os
import time
from langchain_openai import ChatOpenAI

data_sample = data_sample.reset_index(drop=True)

models = [
    'google/gemini-2.5-flash',
    'x-ai/grok-3-mini',
    'openai/gpt-5-mini',
    'google/gemini-2.5-pro',
    'x-ai/grok-4',
    'openai/gpt-5'
]

# Batch inputs 준비
batch_inputs = [
    {
        "problems": row['symptom_topic_filtered_text'],
        "patent_content": row['abstract'] + "\n\n" + row['claim1st']
    }
    for _, row in data_sample.iterrows()
]

# 모델별 처리 (v3 프롬프트로 고정)
for model in models:
    model_name = model.split('/')[-1]
    print(f"\nProcessing {model_name}...")
    
    llm = ChatOpenAI(
        model=model,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    
    # v3 체인 생성 (고정)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    chain_v1 = prompt_v1 | llm | fixing_parser
    
    # Batch 실행
    try:
        results = chain_v1.batch(
            batch_inputs, 
            config={"max_concurrency": 55},
            return_exceptions=True
        )
        
        # 결과 처리 (예외 처리 포함)
        processed_results = []
        for i, resp in enumerate(results):
            if isinstance(resp, Exception):
                print(f"  ✗ Row {i} failed: {str(resp)}")
                processed_results.append({"solutions": [], "error": str(resp)})
            else:
                try:
                    processed_results.append({
                        "solutions": [s.dict() for s in resp.solutions]
                    })
                except AttributeError:
                    processed_results.append({"solutions": [], "error": "Invalid response structure"})
        
        data_sample[model_name] = processed_results
        
        # 통계 출력
        total_solutions = sum(len(r.get('solutions', [])) for r in processed_results)
        error_count = sum(1 for r in processed_results if 'error' in r)
        print(f"  ✓ {model_name} completed: {total_solutions} solutions, {error_count} errors")
        
    except Exception as e:
        print(f"  ✗ {model_name} batch failed: {str(e)}")
        data_sample[model_name] = [{"solutions": [], "error": str(e)}] * len(batch_inputs)
    
    time.sleep(1)

print("\n✓ All models processed!")

# 최종 결과 요약
print("\n=== Final Summary ===")
for model in models:
    model_name = model.split('/')[-1]
    if model_name in data_sample.columns:
        total = sum(len(r.get('solutions', [])) for r in data_sample[model_name])
        errors = sum(1 for r in data_sample[model_name] if 'error' in r)
        print(f"{model_name}: {total} solutions, {errors} errors")

# 데이터 저장
data_sample.to_pickle("./data_solution_results.pkl")
print("\n✓ Results saved to data_solution_results.pkl")

#%% 11. 솔루션 judge
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import json

# Configuration
judge_models = ['google/gemini-2.5-pro', 'openai/gpt-5', 'x-ai/grok-4']
candidate_models = [
    'gemini-2.5-flash', 'grok-3-mini', 'gpt-5-mini',
    'gemini-2.5-pro', 'grok-4', 'gpt-5'
]
permutations = [[0,1,2,3,4,5], [3,4,5,0,1,2]]

MODEL_FAMILIES = {
    'gemini': ['gemini', 'google'],
    'grok': ['grok', 'x-ai'],
    'gpt': ['gpt', 'openai']
}

def get_model_family(model_name):
    model_lower = model_name.lower()
    for family, ids in MODEL_FAMILIES.items():
        if any(i in model_lower for i in ids):
            return family
    return 'unknown'

def filter_candidates(judge_model, all_candidates):
    judge_family = get_model_family(judge_model)
    return [c for c in all_candidates if get_model_family(c) != judge_family]

# Parser & Prompt
parser = JsonOutputParser(pydantic_object=SolutionEvaluation)
prompt = PromptTemplate(
    template=solution_judge_template,
    input_variables=["problems", "patent_content", "candidates"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

for judge_model in judge_models:
    judge_name = judge_model.split('/')[-1]
    print(f"\n=== Processing {judge_name} ===")
    
    valid_candidates = filter_candidates(judge_model, candidate_models)
    print(f"Valid candidates: {valid_candidates}")
    
    llm = ChatOpenAI(
        model=judge_model,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.00,
        max_retries=3
    )
    
    chain = prompt | llm | parser
    
    batch_inputs = []
    row_perm_map = []
    
    for idx, row in data_sample.iterrows():
        problems = row['symptom_topic_filtered_text']
        patent_content = row['abstract'] + "\n\n" + row['claim1st']
        
        if not all(col in row.index and row[col] is not None for col in valid_candidates):
            continue
        
        for perm_idx, perm in enumerate(permutations):
            current_candidates = [valid_candidates[i % len(valid_candidates)] for i in perm[:len(valid_candidates)]]
            extractions = [row[c] for c in current_candidates]
            
            if any(not isinstance(ext, dict) or not ext.get('solutions') for ext in extractions):
                continue
            
            formatted_candidates = "\n".join([
                f"Candidate {i}:\n{json.dumps(ext, indent=2, ensure_ascii=False)}" 
                for i, ext in enumerate(extractions)
            ])
            
            batch_inputs.append({
                "problems": problems,
                "patent_content": patent_content,
                "candidates": formatted_candidates
            })
            row_perm_map.append((idx, perm_idx, perm[:len(valid_candidates)], current_candidates))
    
    print(f"Evaluating {len(batch_inputs)} cases...")
    
    if len(batch_inputs) == 0:
        print("No valid inputs!")
        continue
    
    results = chain.batch(batch_inputs, config={"max_concurrency": 50}, return_exceptions=True)
    
    col_name = f'solution_judge_{judge_name}'
    data_sample[col_name] = [[] for _ in range(len(data_sample))]
    
    error_count = 0
    for (row_idx, perm_idx, perm, current_candidates), result in zip(row_perm_map, results):
        if isinstance(result, Exception):
            error_count += 1
            continue
        
        data_sample.at[row_idx, col_name].append({
            'perm': perm,
            'candidates': current_candidates,
            'scores': result['scores'],
            'best_idx': result['best_candidate_index'],
            'reasoning': result['reasoning']
        })
    
    print(f"✓ Complete ({error_count} errors)")

print("\n✓ All judges complete")
data_sample.to_pickle("./data_solution_6models_eval.pkl")
            


#%% 13. 결과 종합 (Score Aggregation)

data_sample = pd.read_pickle("./data_solution_6models_eval.pkl")

import numpy as np
import pandas as pd

# 모델 및 judge 컬럼 정의
all_models = [
    'gemini-2.5-flash', 'grok-3-mini', 'gpt-5-mini',
    'gemini-2.5-pro', 'grok-4', 'gpt-5'
]

judge_cols = [
    'solution_judge_gemini-2.5-pro', 
    'solution_judge_grok-4',
    'solution_judge_gpt-5'
]

# 결과 초기화
for model in all_models:
    data_sample[f'final_{model}'] = np.nan

print("Aggregating scores...")

# 행별 점수 집계
for idx, row in data_sample.iterrows():
    if idx % 10 == 0:
        print(f"{idx}/{len(data_sample)}", end='\r')
    
    row_model_scores = {m: [] for m in all_models}
    
    for judge_col in judge_cols:
        if judge_col not in data_sample.columns:
            continue
            
        results = row[judge_col]
        
        if not isinstance(results, list) or not results:
            continue
            
        for res in results:
            if not res or 'scores' not in res or 'candidates' not in res or 'perm' not in res:
                continue
                
            scores = res['scores']
            candidates = res['candidates']
            perm = res['perm']
            
            if len(scores) != len(candidates):
                continue
            
            # 순열 역변환
            for pos_idx, score in enumerate(scores):
                if pos_idx < len(perm):
                    original_idx = perm[pos_idx]
                    if original_idx < len(candidates):
                        model_name = candidates[original_idx]
                        
                        if score is not None:
                            try:
                                row_model_scores[model_name].append(float(score))
                            except (ValueError, TypeError):
                                continue
    
    # 평균 계산
    for model in all_models:
        scores = row_model_scores.get(model, [])
        if scores:
            data_sample.at[idx, f'final_{model}'] = np.mean(scores)

print("\n✓ Aggregation complete\n")

# 결과 요약
print("="*80)
print(f"{'Model':<30} {'Valid Samples':<15} {'Mean':<10} {'Median':<10} {'Std':<10}")
print("-"*80)

summary_data = []
for model in all_models:
    col = f'final_{model}'
    valid_scores = data_sample[col].dropna()
    
    if len(valid_scores) > 0:
        summary_data.append({
            'Model': model,
            'Valid Samples': len(valid_scores),
            'Mean': valid_scores.mean(),
            'Median': valid_scores.median(),
            'Std': valid_scores.std()
        })
        print(f"{model:<30} {len(valid_scores):<15} {valid_scores.mean():<10.4f} "
              f"{valid_scores.median():<10.4f} {valid_scores.std():<10.4f}")
    else:
        print(f"{model:<30} {'No data':<15}")

print("="*80)

summary_df = pd.DataFrame(summary_data)

# 결과 저장
data_sample.to_pickle("./data_solution_6models_final.pkl")
print("\n💾 Saved to: data_solution_6models_final.pkl")


#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import numpy as np

# all_models 정의 수정
all_models = [
    'gemini-2.5-flash', 'grok-3-mini', 'gpt-5-mini',
    'gemini-2.5-pro', 'grok-4', 'gpt-5'
]


# ============================================================
# 1. 모델별 비용 정보
# ============================================================
model_costs = {
    'gemini-2.5-flash': 0.30,
    'grok-3-mini': 0.30,
    'gpt-5-mini': 0.25,
    'gemini-2.5-pro': 1.25,
    'gpt-5': 1.25,
    'grok-4': 3.00
}

# ============================================================
# 2. 성능 데이터 수집 (Median 사용)
# ============================================================
model_performance = {}
for model in all_models:
    col = f'final_{model}'
    valid_scores = data_sample[col].dropna()
    
    if len(valid_scores) > 0:
        model_performance[model] = {
            'median': valid_scores.median(),
            'mean': valid_scores.mean(),
            'std': valid_scores.std(),
            'q25': valid_scores.quantile(0.25),
            'q75': valid_scores.quantile(0.75),
            'n': len(valid_scores)
        }

# ============================================================
# 3. 데이터프레임 생성
# ============================================================
lw_models = ['gemini-2.5-flash', 'grok-3-mini', 'gpt-5-mini']
hp_models = ['gemini-2.5-pro', 'grok-4', 'gpt-5']

model_data = []

for model in all_models:
    if model in model_performance:
        model_type = 'High-Performance' if model in hp_models else 'Lightweight'
        
        display_name = model.replace('-', ' ').title()
        display_name = display_name.replace('Gpt', 'GPT').replace('Grok', 'Grok')
        
        model_data.append({
            'model': display_name,
            'model_id': model,
            'cost': model_costs[model],
            'median': model_performance[model]['median'],
            'mean': model_performance[model]['mean'],
            'std': model_performance[model]['std'],
            'q25': model_performance[model]['q25'],
            'q75': model_performance[model]['q75'],
            'iqr': model_performance[model]['q75'] - model_performance[model]['q25'],
            'type': model_type,
            'n_samples': model_performance[model]['n']
        })

model_df = pd.DataFrame(model_data)
model_df = model_df.dropna(subset=['cost', 'median'])
model_df = model_df.reset_index(drop=True)

print("\n" + "="*90)
print("Cost-Performance Analysis (Median Judge Scores)")
print("="*90)
print(f"{'Model':<25} {'Cost':<8} {'Median':<10} {'Mean':<10} {'Std':<10} {'IQR':<10}")
print("-"*90)
for idx, row in model_df.iterrows():
    print(f"{row['model']:<25} ${row['cost']:<7.2f} {row['median']:<10.4f} "
          f"{row['mean']:<10.4f} {row['std']:<10.4f} {row['iqr']:<10.4f}")
print("="*90)

# ============================================================
# 4. X축 Jitter 설정
# ============================================================
cost_groups = model_df.groupby('cost')['model'].apply(list).to_dict()

model_jitter = {}
for cost, models in cost_groups.items():
    if len(models) == 1:
        model_jitter[models[0]] = 1.0
    elif len(models) == 2:
        model_jitter[models[0]] = 0.92
        model_jitter[models[1]] = 1.08
    elif len(models) == 3:
        model_jitter[models[0]] = 0.88
        model_jitter[models[1]] = 1.0
        model_jitter[models[2]] = 1.12

# ============================================================
# 5. 시각화 (Error Bar = IQR)
# ============================================================
plt.figure(figsize=(16, 12))
sns.set_theme(style="whitegrid")

type_colors = {
    'Lightweight': '#2E86AB',
    'High-Performance': '#A23B72'
}

# Scatter plot with IQR as error bars
plotted_types = set()
for i, row in model_df.iterrows():
    jittered_cost = row['cost'] * model_jitter[row['model']]
    color = type_colors[row['type']]
    
    label = row['type'] if row['type'] not in plotted_types else ""
    if label:
        plotted_types.add(row['type'])
    
    # Error bar = IQR (Q1-Q3 범위)
    plt.errorbar(
        jittered_cost,
        row['median'],
        yerr=[[row['median'] - row['q25']], [row['q75'] - row['median']]],
        fmt='o',
        markersize=25,
        capsize=8,
        capthick=2.5,
        elinewidth=2.5,
        alpha=0.85,
        color=color,
        label=label,
        zorder=2
    )

# ============================================================
# 6. 텍스트 라벨
# ============================================================
texts = []
for i, row in model_df.iterrows():
    jittered_cost = row['cost'] * model_jitter[row['model']]
    
    label_text = (f'{row["model"]}\n'
                  f'(Cost: ${row["cost"]:.2f}, '
                  f'Score: {row["median"]:.3f})')
    
    texts.append(plt.text(
        jittered_cost,
        row['median'],
        label_text,
        fontsize=18,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=10
    ))

# ============================================================
# 7. 로그 스케일 및 구분선
# ============================================================
plt.xscale("log")
plt.axvline(1.0, color="gray", linestyle="--", linewidth=2.5, alpha=0.7, 
            label='Cost Threshold ($1)')

# ============================================================
# 8. 축 라벨 및 제목
# ============================================================
plt.title("Model Portfolio: Cost-Performance Trade-off", 
          fontsize=28, pad=20, fontweight='bold')
plt.xlabel("Cost per 1M Input Tokens (USD, log scale)", 
           fontsize=24, labelpad=15)
plt.ylabel("Judge Score (Median, IQR)",
           fontsize=24, labelpad=15)

plt.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)
plt.legend(fontsize=18, title_fontsize=20, markerscale=1.2, 
           loc='lower right', framealpha=0.9)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.xlim(0.20, 4.0)
plt.margins(y=0.1)

# ============================================================
# 9. 라벨 조정
# ============================================================
fig = plt.gcf()
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

bboxes = [t.get_window_extent(renderer=renderer).expanded(1.3, 1.3) 
          for t in texts]

overlap_idx = set()
for i in range(len(bboxes)):
    for j in range(i + 1, len(bboxes)):
        if bboxes[i].overlaps(bboxes[j]):
            overlap_idx.add(i)
            overlap_idx.add(j)

texts_to_adjust = [texts[i] for i in sorted(overlap_idx)]

if texts_to_adjust:
    adjust_text(
        texts_to_adjust,
        only_move={'points': 'xy', 'text': 'xy'},
        autoalign='xy',
        expand_points=(1.6, 1.6),
        expand_text=(1.5, 1.5),
        force_text=3.5,
        force_points=0.8,
        arrowprops=dict(arrowstyle='->', lw=2.0, alpha=0.7, 
                       color='gray', connectionstyle='arc3,rad=0.3')
    )

plt.tight_layout()
plt.savefig('./solution_cost_performance_tradeoff.png', dpi=300, bbox_inches='tight')
plt.show()



#%% 14. GPT-5-mini 채택
import os
import json
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser

# LLM 초기화
llm = ChatOpenAI(
    model='openai/gpt-5-mini',
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
final_chain = prompt_v1 | llm | fixing_parser

# Batch inputs 준비
batch_inputs = [
    {
        "problems": row['symptom_topic_filtered_text'],
        "patent_content": row['abstract'] + "\n\n" + row['claim1st']
    }
    for _, row in data_solution.iterrows()
]

# Batch 실행
print(f"Processing {len(batch_inputs)} solution extractions...")
results = final_chain.batch(
    batch_inputs, 
    config={"max_concurrency": 100}, 
    return_exceptions=True
)

# 결과 변환
final_extractions = []
for result in results:
    if isinstance(result, Exception):
        final_extractions.append({"error": str(result), "solutions": []})
    else:
        final_extractions.append({
            "solutions": [s.dict() for s in result.solutions]
        })

data_solution['gpt-5-mini'] = final_extractions

print(f"✓ Completed: {len(data_solution)}")
print(f"Errors: {sum(1 for x in final_extractions if 'error' in x)}")
print(f"Total solutions: {sum(len(x.get('solutions', [])) for x in final_extractions)}")


#%% ===== 저장 =====
# Pickle 저장
data_solution['final_solution']  = data_solution['gpt-5-mini']
data_solution.to_pickle("./df_with_final_solutions.pkl")

#%%
# CSV 저장 (JSON 컬럼은 문자열로 변환)
data_solution_export = data_solution.copy()
data_solution_export['final_solution'] = data_solution_export['final_solution'].apply(
    lambda x: json.dumps(x, ensure_ascii=False)
)
data_solution_export.to_csv("./data_solution_with_final_solutions.csv", index=False, encoding='utf-8-sig')

print("\n✓ Results saved:")
print("  - data_solution_with_final_solutions.pkl")
print("  - data_solution_with_final_solutions.csv")


