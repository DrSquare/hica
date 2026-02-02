"""
Simple example demonstrating Clio privacy-preserving classification
"""

from clio_pipeline import ClioClassifier


def main():
    print("Clio Privacy-Preserving Hierarchical Classification - Quick Example")
    print("="*70)

    # Sample conversations representing different task types
    conversations = [
        # Programming tasks
        "User: Can you help me implement quicksort in Python?\nAssistant: Sure! Here's a quicksort implementation...",
        "User: Debug this JavaScript error I'm getting.\nAssistant: Let me help you fix that...",
        "User: Write unit tests for this React component.\nAssistant: I'll create comprehensive tests...",

        # Writing tasks
        "User: Help me write a professional email.\nAssistant: Here's a draft email...",
        "User: Create a blog post about AI ethics.\nAssistant: Here's a blog post...",
        "User: Write a product description.\nAssistant: Here's a compelling description...",

        # Research/explanation tasks
        "User: Explain how photosynthesis works.\nAssistant: Photosynthesis is the process...",
        "User: What is machine learning?\nAssistant: Machine learning is a subset of AI...",
        "User: Summarize the theory of relativity.\nAssistant: Einstein's theory of relativity...",

        # Translation tasks
        "User: Translate 'Hello' to French.\nAssistant: 'Bonjour'",
        "User: Convert this Spanish text to English.\nAssistant: Here's the translation...",

        # Data analysis tasks
        "User: Analyze this dataset for trends.\nAssistant: I'll analyze the data...",
        "User: Create a visualization of sales data.\nAssistant: Here's a chart showing...",
    ]

    print(f"\nProcessing {len(conversations)} sample conversations...\n")

    # Initialize Clio classifier
    classifier = ClioClassifier(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        llm_model="gpt-4"  # You can also use "gpt-3.5-turbo" for faster/cheaper processing
    )

    # Run the complete pipeline with privacy barriers
    results = classifier.process_conversations(
        conversations=conversations,
        n_clusters=4,  # Create 4 main clusters
        min_conversations_per_cluster=2,  # Privacy threshold
        min_unique_accounts=1,  # Privacy threshold (adjusted for demo)
        apply_privacy_barriers=True
    )

    # Display key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print(f"Total conversations analyzed: {results['n_conversations']}")
    print(f"Clusters identified: {results['clustering']['n_clusters']}")

    if 'privacy' in results:
        print(f"Clusters removed for privacy: {results['privacy']['n_removed']}")

    print("\nTop-level task categories:")
    if 'hierarchy' in results and 'children' in results['hierarchy']:
        for i, category in enumerate(results['hierarchy']['children'], 1):
            name = category.get('name', 'Unknown')
            n_subclusters = len(category.get('children', []))
            print(f"  {i}. {name} ({n_subclusters} sub-categories)")

    # Save results
    import os
    output_path = os.path.join(os.path.dirname(__file__), "example_results.json")
    classifier.save_results(results, output_path)

    print("\n" + "="*70)
    print("Example complete!")
    print(f"Full results saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
