# Quick Start Guide

Get up and running with Clio privacy-preserving classification in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up OpenAI API Key

Create a `.env` file in the project directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

Get your API key from: https://platform.openai.com/api-keys

## 3. Run the Example

```bash
python example.py
```

This will:
- Process 13 sample conversations
- Extract task facets with privacy preservation
- Cluster similar tasks using all-mpnet-base-v2 embeddings
- Organize into hierarchical categories
- Apply all four privacy barriers
- Save results to `example_results.json`

## 4. Run the Full Demo

```bash
python clio_pipeline.py
```

This runs a more comprehensive demo with 16 conversations.

## 5. Use in Your Code

```python
from clio_pipeline import ClioClassifier

# Initialize
classifier = ClioClassifier(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    llm_model="gpt-4"
)

# Your conversations
conversations = [
    "User: Help me write Python code...",
    "User: Explain quantum physics...",
    # ... more conversations
]

# Process
results = classifier.process_conversations(
    conversations=conversations,
    n_clusters=5,
    apply_privacy_barriers=True
)

# Explore results
print(results["hierarchy"])
```

## Expected Output

The pipeline will show:

1. **Facet Extraction**: Extracting task descriptions with privacy filters
2. **Clustering**: Creating semantic clusters using embeddings
3. **Privacy Filtering**: Removing small clusters below thresholds
4. **Hierarchical Organization**: Building multi-level taxonomy
5. **Privacy Audit**: Final check for private information
6. **Results**: Hierarchical task classification

Example hierarchy:
```
- All Tasks
  - Programming and Software Development
    - Write Python sorting algorithms (n=2)
    - Debug JavaScript code (n=1)
  - Writing and Content Creation
    - Write professional emails (n=1)
    - Create blog posts (n=2)
  - Research and Analysis
    - Explain scientific concepts (n=3)
```

## Key Features

✅ **Privacy-Preserving**: 4-layer privacy architecture
✅ **Hierarchical**: Multi-level task taxonomy
✅ **Semantic**: Uses all-mpnet-base-v2 embeddings
✅ **Flexible**: Configurable clustering and thresholds
✅ **Complete**: End-to-end pipeline with JSON output

## Troubleshooting

### "No module named 'sentence_transformers'"
Run: `pip install sentence-transformers`

### "OpenAI API key not found"
Make sure you have a `.env` file with `OPENAI_API_KEY=...`

### "Rate limit exceeded"
You're hitting OpenAI API limits. Try:
- Using `llm_model="gpt-3.5-turbo"` instead of `gpt-4`
- Processing fewer conversations
- Adding delays between requests

### Model downloads slowly
First run downloads the all-mpnet-base-v2 model (~400MB). This is cached for future runs.

## Cost Estimates

Using GPT-4:
- Facet extraction: ~$0.003 per conversation
- Cluster summaries: ~$0.002 per cluster
- Hierarchy organization: ~$0.01 per run
- Privacy auditing: ~$0.005 per cluster

**Total**: ~$0.05-0.10 per 10 conversations (varies by length)

Using GPT-3.5-Turbo: ~10x cheaper

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check the [paper](https://arxiv.org/abs/2412.13678) for algorithm details
- Explore individual modules: `facet_extraction.py`, `clustering.py`, etc.
- Customize privacy thresholds for your use case
- Integrate with your conversation data

## Need Help?

- Check the paper: https://arxiv.org/abs/2412.13678
- Review module docstrings
- Run individual components with `python module_name.py`
