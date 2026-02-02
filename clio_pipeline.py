"""
Main Pipeline for Clio Privacy-Preserving Hierarchical Classification
Integrates all components: facet extraction, clustering, hierarchy, and privacy barriers
Based on arxiv.org/abs/2412.13678
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from facet_extraction import FacetExtractor
from clustering import SemanticClusterer
from hierarchical_organization import HierarchicalOrganizer
from privacy_barriers import PrivacyBarriers


class ClioClassifier:
    """
    Complete Clio pipeline for privacy-preserving hierarchical task classification.
    """

    def __init__(self,
                 openai_api_key: str = None,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 llm_model: str = "gpt-4"):
        """
        Initialize the Clio classifier.

        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            embedding_model: Sentence transformer model for embeddings
            llm_model: OpenAI model for LLM operations
        """
        self.llm_model = llm_model

        print("Initializing Clio Privacy-Preserving Classifier...")
        print(f"  Embedding model: {embedding_model}")
        print(f"  LLM model: {llm_model}")

        self.facet_extractor = FacetExtractor(api_key=openai_api_key)
        self.clusterer = SemanticClusterer(embedding_model=embedding_model)
        self.organizer = HierarchicalOrganizer(api_key=openai_api_key)
        self.privacy_barriers = PrivacyBarriers(api_key=openai_api_key)

        print("Initialization complete!\n")

    def process_conversations(self,
                             conversations: List[str],
                             n_clusters: int = None,
                             min_clusters: int = 5,
                             max_clusters: int = 50,
                             min_conversations_per_cluster: int = 3,
                             min_unique_accounts: int = 2,
                             target_hierarchy_levels: int = 2,
                             apply_privacy_barriers: bool = True) -> Dict[str, Any]:
        """
        Complete Clio pipeline: process conversations with privacy preservation.

        Args:
            conversations: List of conversation texts
            n_clusters: Number of clusters (None for auto-detection)
            min_clusters: Minimum clusters if auto-detecting
            max_clusters: Maximum clusters if auto-detecting
            min_conversations_per_cluster: Min conversations for cluster threshold
            min_unique_accounts: Min unique accounts for cluster threshold
            target_hierarchy_levels: Number of hierarchical levels
            apply_privacy_barriers: Whether to apply privacy barriers

        Returns:
            Dictionary containing all pipeline results
        """
        print("="*70)
        print("CLIO PRIVACY-PRESERVING HIERARCHICAL CLASSIFICATION PIPELINE")
        print("="*70)
        print(f"\nProcessing {len(conversations)} conversations...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_conversations": len(conversations),
            "config": {
                "n_clusters": n_clusters,
                "min_clusters": min_clusters,
                "max_clusters": max_clusters,
                "min_conversations_per_cluster": min_conversations_per_cluster,
                "min_unique_accounts": min_unique_accounts,
                "llm_model": self.llm_model
            }
        }

        # Step 1: Extract facets (with privacy layer 1)
        print("\n" + "="*70)
        print("STEP 1: FACET EXTRACTION")
        print("="*70)
        facets_list = self.facet_extractor.batch_extract_facets(
            conversations,
            facet_names=["task"],  # Focus on task facet for classification
            model=self.llm_model
        )
        results["facets"] = facets_list

        # Extract task descriptions for clustering
        task_descriptions = [facets["task"] for facets in facets_list]

        # Step 2: Embed and cluster
        print("\n" + "="*70)
        print("STEP 2: SEMANTIC CLUSTERING")
        print("="*70)
        clustering_results = self.clusterer.fit_cluster(
            task_descriptions,
            n_clusters=n_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            create_umap=True
        )
        results["clustering"] = {
            "n_clusters": clustering_results["n_clusters"],
            "cluster_labels": clustering_results["cluster_labels"].tolist(),
            "cluster_sizes": [len(clustering_results["clusters"][i])
                            for i in sorted(clustering_results["clusters"].keys())]
        }

        # Step 3: Apply privacy barrier 2 (aggregation threshold)
        if apply_privacy_barriers:
            print("\n" + "="*70)
            print("STEP 3: PRIVACY BARRIER - AGGREGATION THRESHOLD")
            print("="*70)
            filtered_clusters = self.privacy_barriers.barrier_2_aggregation_threshold(
                clustering_results["clusters"],
                min_conversations=min_conversations_per_cluster,
                min_unique_accounts=min_unique_accounts
            )
            clustering_results["clusters"] = filtered_clusters
            results["clustering"]["n_clusters_after_filter"] = len(filtered_clusters)

        # Step 4: Generate cluster summaries and organize hierarchy
        print("\n" + "="*70)
        print("STEP 4: HIERARCHICAL ORGANIZATION")
        print("="*70)
        hierarchy = self.organizer.process_clusters(
            clustering_results["clusters"],
            model=self.llm_model,
            privacy_mode=apply_privacy_barriers
        )
        results["hierarchy"] = hierarchy

        # Step 5: Apply privacy barrier 4 (auditing)
        if apply_privacy_barriers:
            print("\n" + "="*70)
            print("STEP 5: PRIVACY BARRIER - AUDIT")
            print("="*70)
            audited_hierarchy, removed_clusters = self.privacy_barriers.barrier_4_auditing(
                hierarchy,
                model=self.llm_model
            )
            results["hierarchy"] = audited_hierarchy
            results["privacy"] = {
                "removed_clusters": removed_clusters,
                "n_removed": len(removed_clusters)
            }

        # Print final results
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"\nProcessed {len(conversations)} conversations")
        print(f"Generated {results['clustering']['n_clusters']} clusters")
        if apply_privacy_barriers and "n_clusters_after_filter" in results["clustering"]:
            print(f"After aggregation filter: {results['clustering']['n_clusters_after_filter']} clusters")
        if apply_privacy_barriers and "privacy" in results:
            print(f"Removed in audit: {results['privacy']['n_removed']} clusters")

        print("\nHierarchical Classification:")
        self.organizer.print_hierarchy(results["hierarchy"])

        return results

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save pipeline results to JSON file.

        Args:
            results: Results dictionary from process_conversations
            output_path: Path to save JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=convert_numpy)

        print(f"\nResults saved to: {output_path}")


def main():
    """
    Example usage of the Clio classifier with sample conversations.
    """
    # Sample conversations from the appendix domain (coding, writing, research)
    sample_conversations = [
        # Coding tasks
        """User: Can you help me write a Python function to implement bubble sort?
        Assistant: Sure! Here's a bubble sort implementation...""",

        """User: I need to debug this JavaScript code that's throwing errors.
        Assistant: Let me help you identify the issue...""",

        """User: How do I implement a binary search tree in Java?
        Assistant: I'll show you a BST implementation...""",

        """User: Write a function to reverse a linked list in C++.
        Assistant: Here's how to reverse a linked list...""",

        # Writing tasks
        """User: Help me write a professional email to my manager about taking time off.
        Assistant: I'll help you draft that email...""",

        """User: Can you write a blog post about sustainable living?
        Assistant: Here's a blog post on sustainable living...""",

        """User: I need help writing a cover letter for a software engineering position.
        Assistant: Let me help you create a compelling cover letter...""",

        """User: Write a short story about a time traveler.
        Assistant: Here's a creative short story...""",

        # Research tasks
        """User: Explain the concept of quantum entanglement.
        Assistant: Quantum entanglement is a phenomenon...""",

        """User: What are the causes of climate change?
        Assistant: Climate change is primarily caused by...""",

        """User: Can you summarize recent advances in gene therapy?
        Assistant: Recent advances in gene therapy include...""",

        """User: Explain how neural networks work.
        Assistant: Neural networks are computational models...""",

        # Translation tasks
        """User: Translate this English text to Spanish: 'Hello, how are you?'
        Assistant: 'Hola, ¿cómo estás?'""",

        """User: Can you translate this French paragraph to English?
        Assistant: Here's the English translation...""",

        # More coding tasks
        """User: Help me optimize this SQL query for better performance.
        Assistant: Let me suggest some optimizations...""",

        """User: Write unit tests for this Python function.
        Assistant: Here are comprehensive unit tests...""",
    ]

    # Initialize classifier
    classifier = ClioClassifier(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        llm_model="gpt-4"
    )

    # Process conversations
    results = classifier.process_conversations(
        conversations=sample_conversations,
        n_clusters=5,  # Specify number of clusters
        min_conversations_per_cluster=2,  # Lower threshold for demo
        min_unique_accounts=1,  # Lower threshold for demo
        apply_privacy_barriers=True
    )

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "clio_results.json")
    classifier.save_results(results, output_path)

    print("\n" + "="*70)
    print("Example complete! Check clio_results.json for full output.")
    print("="*70)


if __name__ == "__main__":
    main()
