# Clio: Privacy-Preserving Hierarchical Task Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.13678-b31b1b.svg)](https://arxiv.org/abs/2412.13678)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the privacy-preserving hierarchical task/intent classification algorithm from the paper:
**"Clio: Privacy-Preserving Insights into Real-World AI Use"** ([arxiv.org/abs/2412.13678](https://arxiv.org/abs/2412.13678))

## Overview

This implementation provides a complete pipeline for analyzing and classifying conversations while preserving privacy through multiple barriers. The system:

1. Extracts task facets from conversations without exposing private information
2. Clusters similar tasks using semantic embeddings
3. Organizes clusters into a hierarchical taxonomy
4. Applies four sequential privacy barriers to ensure no private data reaches analysts

## Algorithm Components

### 1. Facet Extraction (`facet_extraction.py`)
Extracts four key facets from conversations:
- **Request**: The user's overall request
- **Task**: The specific task being performed
- **Language**: Languages present in the conversation
- **Concern Level**: Safety rating (1-5 scale)

All extraction includes privacy-preserving prompts that explicitly instruct the LLM to omit personally identifiable information.

### 2. Semantic Clustering (`clustering.py`)
- Uses **all-mpnet-base-v2** sentence transformer (as specified in the paper)
- Generates normalized embeddings for semantic similarity
- Performs k-means clustering with automatic cluster count determination
- Creates UMAP 2D visualizations for exploration

### 3. Hierarchical Organization (`hierarchical_organization.py`)
- Generates descriptive summaries for each cluster
- Organizes base-level clusters into multi-level hierarchies
- Uses LLM prompting to create meaningful category structures
- Enables navigation from high-level patterns to granular task types

### 4. Privacy Barriers (`privacy_barriers.py`)
Implements the four sequential privacy layers from the paper:

**Layer 1: Conversation Summarization**
- Extracts attributes while omitting private information
- Implemented in facet extraction prompts

**Layer 2: Cluster Aggregation Thresholds**
- Discards clusters with insufficient unique accounts/conversations
- Prevents identification through small sample sizes

**Layer 3: Cluster Summary Generation**
- Explicit instructions to exclude private data
- Sanitizes summaries using privacy-aware LLM prompts

**Layer 4: Cluster Auditing**
- Model-based review using regex patterns and LLM checks
- Removes any clusters containing private information
- Detects: emails, phone numbers, SSNs, credit cards, IPs, etc.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- sentence-transformers==2.3.1
- scikit-learn==1.4.0
- numpy==1.26.3
- openai==1.12.0
- python-dotenv==1.0.1
- umap-learn==0.5.5
- matplotlib==3.8.2

## Configuration

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from clio_pipeline import ClioClassifier

# Initialize classifier
classifier = ClioClassifier(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    llm_model="gpt-4"
)

# Process conversations
conversations = [
    "User: Help me write a Python sorting function...",
    "User: Explain quantum mechanics...",
    # ... more conversations
]

results = classifier.process_conversations(
    conversations=conversations,
    n_clusters=5,
    min_conversations_per_cluster=3,
    min_unique_accounts=2,
    apply_privacy_barriers=True
)

# Save results
classifier.save_results(results, "output.json")
```

### Running the Example

```bash
python clio_pipeline.py
```

This will process sample conversations and generate:
- `clio_results.json`: Complete pipeline results
- Console output showing the hierarchical classification

### Individual Components

#### Facet Extraction Only
```python
from facet_extraction import FacetExtractor

extractor = FacetExtractor()
facets = extractor.extract_all_facets(conversation_text)
print(facets["task"])
```

#### Clustering Only
```python
from clustering import SemanticClusterer

clusterer = SemanticClusterer()
results = clusterer.fit_cluster(
    texts=task_descriptions,
    n_clusters=10
)
```

#### Hierarchical Organization Only
```python
from hierarchical_organization import HierarchicalOrganizer

organizer = HierarchicalOrganizer()
hierarchy = organizer.process_clusters(clusters)
organizer.print_hierarchy(hierarchy)
```

#### Privacy Barriers Only
```python
from privacy_barriers import PrivacyBarriers

barriers = PrivacyBarriers()
has_issues, issues = barriers.detect_privacy_leakage(text)
```

## Key Parameters

### ClioClassifier.process_conversations()

- `n_clusters` (int): Number of clusters (None for auto-detection)
- `min_clusters` (int): Minimum clusters when auto-detecting (default: 5)
- `max_clusters` (int): Maximum clusters when auto-detecting (default: 50)
- `min_conversations_per_cluster` (int): Privacy threshold (default: 3)
- `min_unique_accounts` (int): Privacy threshold (default: 2)
- `target_hierarchy_levels` (int): Hierarchy depth (default: 2)
- `apply_privacy_barriers` (bool): Enable privacy layers (default: True)

## Output Structure

The pipeline returns a dictionary containing:

```python
{
    "timestamp": "2024-...",
    "n_conversations": 100,
    "config": {...},
    "facets": [...],  # Extracted facets for each conversation
    "clustering": {
        "n_clusters": 10,
        "cluster_labels": [...],
        "cluster_sizes": [...]
    },
    "hierarchy": {
        "name": "All Tasks",
        "children": [
            {
                "name": "Programming and Software Development",
                "children": [...]
            },
            ...
        ]
    },
    "privacy": {
        "removed_clusters": [...],
        "n_removed": 2
    }
}
```

## Embedding Model: all-mpnet-base-v2

As specified in the Clio paper, this implementation uses the **all-mpnet-base-v2** sentence transformer model:

- Based on MPNet architecture
- Trained on 1B+ training pairs
- 768-dimensional embeddings
- Optimized for semantic similarity tasks
- Publicly available via HuggingFace

## Sample Prompts

The implementation includes privacy-preserving prompts based on the paper's approach:

### Task Extraction
```
What task is the model being asked to perform?
Focus on the general task type without including private information,
personal details, or sensitive content.
```

### Concern Level Rating
```
Rate the safety concern level of this conversation on a 1-5 scale:
1 - Not concerning at all. Completely benign conversation.
5 - Extremely concerning. Immediate review needed.
```

### Cluster Summary Generation
```
Create a short, specific cluster name that captures the task.
The cluster name should be a clear single sentence that captures
the specific action or task.
```

## Privacy Guarantees

This implementation follows the Clio paper's privacy architecture:

1. **No raw conversation access**: Only anonymized facets are used
2. **Aggregation thresholds**: Small clusters are filtered out
3. **Explicit sanitization**: LLM-based privacy cleaning
4. **Final audit**: Multi-layer detection of privacy leaks

The system is designed to achieve 94%+ accuracy on task classification while maintaining "very low levels" of private information exposure (as demonstrated in the paper's evaluation).

## References

- Paper: [Clio: Privacy-Preserving Insights into Real-World AI Use](https://arxiv.org/abs/2412.13678)
- Anthropic Research: [https://www.anthropic.com/research/clio](https://www.anthropic.com/research/clio)
- Open Source Implementation: [https://github.com/Phylliida/OpenClio](https://github.com/Phylliida/OpenClio)
- all-mpnet-base-v2: [HuggingFace Model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

## License

This implementation is provided for research and educational purposes based on the publicly available Clio paper.

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{clio2024,
  title={Clio: Privacy-Preserving Insights into Real-World AI Use},
  author={[Authors from paper]},
  journal={arXiv preprint arXiv:2412.13678},
  year={2024}
}
```
