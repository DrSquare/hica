"""
Hierarchical Organization Module for Clio Privacy-Preserving Classification
Organizes clusters into multi-level hierarchies using LLM prompts
Based on arxiv.org/abs/2412.13678
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()


class HierarchicalOrganizer:
    """
    Organizes base-level clusters into a multi-level hierarchy.
    Uses LLM prompting to generate cluster summaries and organize them hierarchically.
    """

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_cluster_summary(self,
                                cluster_items: List[str],
                                model: str = "gpt-4",
                                privacy_mode: bool = True) -> Dict[str, str]:
        """
        Generate a summary name and description for a cluster.

        Args:
            cluster_items: List of text items in the cluster
            model: OpenAI model to use
            privacy_mode: Whether to enforce privacy constraints

        Returns:
            Dictionary with 'name' and 'description' keys
        """
        privacy_instruction = ""
        if privacy_mode:
            privacy_instruction = """
CRITICAL PRIVACY REQUIREMENT:
- Do NOT include any personally identifiable information
- Do NOT mention specific names, locations, companies, or identifying details
- Focus only on the general task or theme
- Use abstract, generalized descriptions
"""

        system_prompt = f"""You are analyzing clusters of conversation tasks to create clear, concise summaries.

{privacy_instruction}

Your task:
1. Identify the common theme or task type across all items
2. Create a short, specific cluster name (max 8 words) that captures the task
3. Write a one-sentence description explaining what these tasks involve

Guidelines for cluster names:
- Use clear, action-oriented language
- Be specific about the task type
- Examples: "Write Python sorting algorithms", "Explain quantum physics concepts", "Translate between European languages"
"""

        items_text = "\n".join([f"- {item}" for item in cluster_items[:20]])  # Limit to 20 items
        if len(cluster_items) > 20:
            items_text += f"\n... and {len(cluster_items) - 20} more similar items"

        user_prompt = f"""Analyze these {len(cluster_items)} similar task descriptions and create a summary:

{items_text}

Provide your response in this exact format:
NAME: [cluster name]
DESCRIPTION: [one sentence description]"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            content = response.choices[0].message.content.strip()

            # Parse the response
            lines = content.split('\n')
            name = ""
            description = ""

            for line in lines:
                if line.startswith("NAME:"):
                    name = line.replace("NAME:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()

            if not name or not description:
                # Fallback parsing
                name = lines[0].replace("NAME:", "").strip()
                description = lines[1].replace("DESCRIPTION:", "").strip() if len(lines) > 1 else "Task cluster"

            return {
                "name": name,
                "description": description
            }

        except Exception as e:
            print(f"Error generating cluster summary: {str(e)}")
            return {
                "name": f"Task Cluster (n={len(cluster_items)})",
                "description": "A cluster of similar tasks"
            }

    def organize_into_hierarchy(self,
                                cluster_summaries: List[Dict[str, Any]],
                                target_levels: int = 2,
                                model: str = "gpt-4") -> Dict[str, Any]:
        """
        Organize flat clusters into a hierarchical structure.

        Args:
            cluster_summaries: List of cluster summary dictionaries
            target_levels: Number of hierarchical levels to create
            model: OpenAI model to use

        Returns:
            Hierarchical structure as nested dictionaries
        """
        if len(cluster_summaries) <= 5:
            # Too few clusters for meaningful hierarchy
            return {
                "level": 0,
                "name": "All Tasks",
                "children": cluster_summaries
            }

        print(f"Organizing {len(cluster_summaries)} clusters into {target_levels}-level hierarchy...")

        # Create summary list for LLM
        summaries_text = "\n".join([
            f"{i+1}. {item['name']}: {item['description']}"
            for i, item in enumerate(cluster_summaries)
        ])

        system_prompt = """You are organizing task clusters into a hierarchical taxonomy.

Your goal is to:
1. Identify high-level categories that group related tasks
2. Assign each numbered cluster to the most appropriate high-level category
3. Create 4-8 high-level categories that cover all clusters

Use clear, general category names like:
- "Programming and Software Development"
- "Writing and Content Creation"
- "Research and Analysis"
- "Translation and Language Tasks"
- "Education and Learning"
"""

        user_prompt = f"""Organize these {len(cluster_summaries)} task clusters into high-level categories:

{summaries_text}

Provide your response in this format:
CATEGORY: [Category Name]
DESCRIPTION: [One sentence description]
CLUSTERS: [comma-separated list of cluster numbers, e.g., 1, 3, 5]

[Repeat for each category]"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            # Parse the hierarchical structure
            hierarchy = {
                "level": 0,
                "name": "All Tasks",
                "description": "Root level containing all task categories",
                "children": []
            }

            current_category = None
            lines = content.split('\n')

            for line in lines:
                line = line.strip()
                if line.startswith("CATEGORY:"):
                    if current_category:
                        hierarchy["children"].append(current_category)
                    current_category = {
                        "level": 1,
                        "name": line.replace("CATEGORY:", "").strip(),
                        "description": "",
                        "children": []
                    }
                elif line.startswith("DESCRIPTION:") and current_category:
                    current_category["description"] = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("CLUSTERS:") and current_category:
                    cluster_nums = line.replace("CLUSTERS:", "").strip()
                    # Parse cluster numbers
                    try:
                        nums = [int(n.strip()) - 1 for n in cluster_nums.split(',') if n.strip().isdigit()]
                        for num in nums:
                            if 0 <= num < len(cluster_summaries):
                                cluster_copy = cluster_summaries[num].copy()
                                cluster_copy["level"] = 2
                                current_category["children"].append(cluster_copy)
                    except ValueError:
                        pass

            # Add last category
            if current_category:
                hierarchy["children"].append(current_category)

            # Handle any unassigned clusters
            assigned_indices = set()
            for category in hierarchy["children"]:
                for child in category["children"]:
                    if "cluster_id" in child:
                        assigned_indices.add(child["cluster_id"])

            unassigned = [
                cluster for i, cluster in enumerate(cluster_summaries)
                if cluster.get("cluster_id", i) not in assigned_indices
            ]

            if unassigned:
                other_category = {
                    "level": 1,
                    "name": "Other Tasks",
                    "description": "Miscellaneous tasks that don't fit into other categories",
                    "children": [dict(c, level=2) for c in unassigned]
                }
                hierarchy["children"].append(other_category)

            print(f"Created hierarchy with {len(hierarchy['children'])} top-level categories")

            return hierarchy

        except Exception as e:
            print(f"Error organizing hierarchy: {str(e)}")
            # Fallback: create simple flat hierarchy
            return {
                "level": 0,
                "name": "All Tasks",
                "description": "Root level containing all tasks",
                "children": [dict(c, level=1) for c in cluster_summaries]
            }

    def process_clusters(self,
                        clusters: Dict[int, List[Dict[str, Any]]],
                        model: str = "gpt-4",
                        privacy_mode: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: generate summaries for all clusters and organize hierarchically.

        Args:
            clusters: Dictionary mapping cluster IDs to lists of items
            model: OpenAI model to use
            privacy_mode: Whether to enforce privacy constraints

        Returns:
            Hierarchical structure with cluster summaries
        """
        print(f"\nGenerating summaries for {len(clusters)} clusters...")

        cluster_summaries = []
        for cluster_id, items in sorted(clusters.items()):
            print(f"  Processing cluster {cluster_id} ({len(items)} items)...")

            # Extract text from items
            texts = [item['text'] if isinstance(item, dict) else str(item) for item in items]

            # Generate summary
            summary = self.generate_cluster_summary(texts, model, privacy_mode)
            summary["cluster_id"] = cluster_id
            summary["size"] = len(items)
            summary["items"] = texts[:10]  # Keep first 10 items as examples

            cluster_summaries.append(summary)

        # Organize into hierarchy
        print("\nOrganizing clusters into hierarchy...")
        hierarchy = self.organize_into_hierarchy(cluster_summaries, target_levels=2, model=model)

        return hierarchy

    def print_hierarchy(self, hierarchy: Dict[str, Any], indent: int = 0):
        """
        Pretty print the hierarchical structure.

        Args:
            hierarchy: Hierarchical structure dictionary
            indent: Current indentation level
        """
        prefix = "  " * indent

        # Print current node
        name = hierarchy.get("name", "Unnamed")
        size = hierarchy.get("size", "")
        size_str = f" (n={size})" if size else ""

        print(f"{prefix}- {name}{size_str}")

        if "description" in hierarchy and hierarchy["description"]:
            print(f"{prefix}  {hierarchy['description']}")

        # Print children recursively
        if "children" in hierarchy:
            for child in hierarchy["children"]:
                self.print_hierarchy(child, indent + 1)


if __name__ == "__main__":
    # Example usage
    organizer = HierarchicalOrganizer()

    # Sample clusters
    sample_clusters = {
        0: [
            {"text": "The task is to write Python code for sorting algorithms"},
            {"text": "The task is to implement a sorting function in Python"}
        ],
        1: [
            {"text": "The task is to explain quantum mechanics concepts"},
            {"text": "The task is to describe quantum physics principles"}
        ],
        2: [
            {"text": "The task is to translate English text to Spanish"},
            {"text": "The task is to translate French text to German"}
        ],
        3: [
            {"text": "The task is to write a blog post about cooking"},
            {"text": "The task is to create content about recipes and food"}
        ]
    }

    print("Processing clusters and creating hierarchy...")
    hierarchy = organizer.process_clusters(sample_clusters, model="gpt-4")

    print("\n" + "="*60)
    print("HIERARCHICAL ORGANIZATION:")
    print("="*60)
    organizer.print_hierarchy(hierarchy)
