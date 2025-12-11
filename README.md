# Prompts for "Identifying Technological Problems and Exploring Potential Solutions  to Support R&amp;D : Tech-mining with multi-LLM Applications"

This repository contains the prompts used in the paper "Identifying Technological Problems and Exploring Potential Solutions  to Support R&amp;D : Tech-mining with multi-LLM Applications"
We aim to ensure reproducibility and facilitate further research by providing the exact prompts used in our experiments.

## Task Description
We conducted experiments on four specific tasks related to technical problem analysis:

1.  **Paragraph Filtering**
2.  **Target-Symptom Extraction**
3.  **Inter-problem Causal Analysis**
4.  **Technical Solution Identification**

## Prompt Variations 
For each task, we designed four variations of prompts to analyze the impact of different prompt components.

The variations are as follows:

| Variation | Filename | Description |

| **1. Baseline** | `01_baseline.md` | Basic structure of prompt |

| **2. Positive examples** | `02_pos_examples.md` | Few shot(k=3) examples |

| **3. Negative examples** | `03_neg_examples.md` | Negative examples |

| **4. Definition** | `04_definition.md` | Explicit definitions of tasks |

## Repository Structure
```text
.
├── LICENSE
├── README.md
│
├── 01_Paragraph_Filtering/
│   ├── 01_baseline.md
│   ├── 02_pos_examples.md
│   ├── 03_neg_examples.md
│   └── 04_definition.md
│
├── 02_Target_Symptom_Extraction/
│   ├── 01_baseline.md
│   ├── ...
│
├── 03_Inter_Problem_Causal_Analysis/
│   ├── ...
│
└── 04_Technical_Solution_Identification/
    ├── ...
```

## Usage
Each .md file contains the full prompt text used for the LLM input. You can copy the content directly to reproduce our experiments.

Placeholders: If the prompts contain placeholders like {{definition_block}}, please replace them with your actual test data.

## License
This project is licensed under the MIT License - see the LICENSE file for details
