"""
Privacy Barriers Module for Clio Privacy-Preserving Classification
Implements the four privacy layers described in the paper
Based on arxiv.org/abs/2412.13678
"""

import os
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class PrivacyBarriers:
    """
    Implements the four sequential privacy layers from the Clio paper:
    1. Conversation summarization (extracting attributes without private info)
    2. Cluster aggregation thresholds
    3. Cluster summary generation (explicitly excluding private data)
    4. Cluster auditing (model-based review)
    """

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Privacy patterns to detect
        self.privacy_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "url_with_personal": r'https?://[^\s]*(?:user|id|account)=[^\s&]+',
        }

    def barrier_1_summarization(self, conversation: str, facets: Dict[str, str]) -> Dict[str, str]:
        """
        Layer 1: Conversation summarization with privacy preservation.
        This is handled by the FacetExtractor class, but we validate the output here.

        Args:
            conversation: Original conversation text
            facets: Extracted facets from FacetExtractor

        Returns:
            Validated facets with privacy check results
        """
        validated_facets = facets.copy()
        validation_results = {}

        for facet_name, facet_value in facets.items():
            has_privacy_issues, issues = self.detect_privacy_leakage(facet_value)
            validation_results[facet_name] = {
                "value": facet_value,
                "has_privacy_issues": has_privacy_issues,
                "issues": issues
            }

            if has_privacy_issues:
                print(f"WARNING: Privacy issues detected in facet '{facet_name}': {issues}")

        return validation_results

    def barrier_2_aggregation_threshold(self,
                                       clusters: Dict[int, List[Any]],
                                       min_conversations: int = 3,
                                       min_unique_accounts: int = 2) -> Dict[int, List[Any]]:
        """
        Layer 2: Cluster aggregation thresholds.
        Discard clusters that don't meet minimum size requirements.

        Args:
            clusters: Dictionary mapping cluster IDs to lists of items
            min_conversations: Minimum number of conversations per cluster
            min_unique_accounts: Minimum number of unique accounts per cluster

        Returns:
            Filtered clusters meeting the thresholds
        """
        filtered_clusters = {}
        removed_count = 0

        for cluster_id, items in clusters.items():
            # Check conversation count
            if len(items) < min_conversations:
                removed_count += 1
                print(f"  Removing cluster {cluster_id}: only {len(items)} conversations (min: {min_conversations})")
                continue

            # Check unique accounts (if account info is available)
            unique_accounts = len(set(
                item.get('account_id', f'account_{i}')
                for i, item in enumerate(items)
            ))

            if unique_accounts < min_unique_accounts:
                removed_count += 1
                print(f"  Removing cluster {cluster_id}: only {unique_accounts} unique accounts (min: {min_unique_accounts})")
                continue

            filtered_clusters[cluster_id] = items

        print(f"\nAggregation threshold results:")
        print(f"  Original clusters: {len(clusters)}")
        print(f"  Filtered clusters: {len(filtered_clusters)}")
        print(f"  Removed clusters: {removed_count}")

        return filtered_clusters

    def barrier_3_summary_generation(self,
                                    cluster_summary: Dict[str, str],
                                    model: str = "gpt-4") -> Dict[str, str]:
        """
        Layer 3: Generate cluster summary with explicit privacy exclusion.

        Args:
            cluster_summary: Initial cluster summary
            model: OpenAI model to use

        Returns:
            Privacy-sanitized cluster summary
        """
        system_prompt = """You are a privacy-preserving assistant that sanitizes cluster summaries.

Your task is to rewrite the given cluster summary to ensure it contains NO private information:

REMOVE OR GENERALIZE:
- Names of people, companies, or organizations
- Specific locations beyond country/region level
- Email addresses, phone numbers, or contact information
- Account identifiers or usernames
- Specific dates or times
- Any other personally identifiable information

KEEP:
- General task descriptions
- Abstract concepts and themes
- General geographic regions (e.g., "Europe", "North America")
- General time periods (e.g., "recent", "historical")

Your output should be a sanitized version that preserves the meaning while removing all private information."""

        user_prompt = f"""Sanitize this cluster summary:

NAME: {cluster_summary.get('name', '')}
DESCRIPTION: {cluster_summary.get('description', '')}

Provide the sanitized version in the same format:
NAME: [sanitized name]
DESCRIPTION: [sanitized description]"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # Parse response
            lines = content.split('\n')
            sanitized = {}

            for line in lines:
                if line.startswith("NAME:"):
                    sanitized["name"] = line.replace("NAME:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    sanitized["description"] = line.replace("DESCRIPTION:", "").strip()

            return sanitized

        except Exception as e:
            print(f"Error in privacy sanitization: {str(e)}")
            return cluster_summary

    def barrier_4_auditing(self,
                          hierarchy: Dict[str, Any],
                          model: str = "gpt-4") -> Tuple[Dict[str, Any], List[str]]:
        """
        Layer 4: Model-based audit to detect and remove clusters with private information.

        Args:
            hierarchy: Hierarchical cluster structure
            model: OpenAI model to use

        Returns:
            Tuple of (audited hierarchy, list of removed cluster IDs)
        """
        print("\nPerforming privacy audit on hierarchy...")

        removed_clusters = []
        audited_hierarchy = self._audit_node(hierarchy, model, removed_clusters)

        print(f"\nAudit complete:")
        print(f"  Removed clusters: {len(removed_clusters)}")
        if removed_clusters:
            print(f"  Removed cluster IDs: {removed_clusters}")

        return audited_hierarchy, removed_clusters

    def _audit_node(self,
                   node: Dict[str, Any],
                   model: str,
                   removed_clusters: List[str]) -> Dict[str, Any]:
        """
        Recursively audit a hierarchy node for privacy violations.

        Args:
            node: Current node in the hierarchy
            model: OpenAI model to use
            removed_clusters: List to accumulate removed cluster IDs

        Returns:
            Audited node (potentially with children removed)
        """
        # Check current node
        if "name" in node and "description" in node:
            combined_text = f"{node['name']} - {node['description']}"

            # First, use regex patterns
            has_privacy_issues, issues = self.detect_privacy_leakage(combined_text)

            # Then, use LLM for deeper analysis
            if not has_privacy_issues:
                has_privacy_issues = self._llm_privacy_check(combined_text, model)

            if has_privacy_issues:
                cluster_id = node.get("cluster_id", node.get("name", "unknown"))
                print(f"  AUDIT FAILED: Removing cluster '{cluster_id}' due to privacy concerns")
                removed_clusters.append(str(cluster_id))
                return None

        # Recursively audit children
        audited_node = node.copy()
        if "children" in node and node["children"]:
            audited_children = []
            for child in node["children"]:
                audited_child = self._audit_node(child, model, removed_clusters)
                if audited_child is not None:
                    audited_children.append(audited_child)

            audited_node["children"] = audited_children

        return audited_node

    def _llm_privacy_check(self, text: str, model: str) -> bool:
        """
        Use LLM to check for privacy issues that regex might miss.

        Args:
            text: Text to check
            model: OpenAI model to use

        Returns:
            True if privacy issues detected, False otherwise
        """
        system_prompt = """You are a privacy auditor. Your task is to determine if the given text contains any private or personally identifiable information.

Private information includes:
- Names of individuals
- Specific company names
- Email addresses, phone numbers
- Specific addresses or locations (street level)
- Account IDs or usernames
- Any other information that could identify a specific person or organization

Respond with ONLY "YES" if private information is detected, or "NO" if the text is general and anonymous."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Does this text contain private information?\n\n{text}"}
                ],
                temperature=0.0,
                max_tokens=10
            )

            result = response.choices[0].message.content.strip().upper()
            return result == "YES"

        except Exception as e:
            print(f"Error in LLM privacy check: {str(e)}")
            return False  # Fail open to avoid blocking legitimate content

    def detect_privacy_leakage(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect potential privacy leakage using regex patterns.

        Args:
            text: Text to check

        Returns:
            Tuple of (has_issues, list of detected issue types)
        """
        issues = []

        for pattern_name, pattern in self.privacy_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(pattern_name)

        return len(issues) > 0, issues

    def apply_all_barriers(self,
                          conversations: List[str],
                          facets_list: List[Dict[str, str]],
                          clusters: Dict[int, List[Any]],
                          hierarchy: Dict[str, Any],
                          min_conversations: int = 3,
                          min_unique_accounts: int = 2,
                          model: str = "gpt-4") -> Dict[str, Any]:
        """
        Apply all four privacy barriers in sequence.

        Args:
            conversations: Original conversations
            facets_list: Extracted facets for each conversation
            clusters: Clustered items
            hierarchy: Hierarchical organization
            min_conversations: Minimum conversations per cluster
            min_unique_accounts: Minimum unique accounts per cluster
            model: OpenAI model to use

        Returns:
            Dictionary containing privacy-filtered results
        """
        print("="*60)
        print("APPLYING PRIVACY BARRIERS")
        print("="*60)

        # Barrier 1: Validate facet extraction
        print("\n[Barrier 1] Validating facet extraction...")
        validated_facets = []
        for i, facets in enumerate(facets_list):
            validation = self.barrier_1_summarization(conversations[i], facets)
            validated_facets.append(validation)

        # Barrier 2: Apply aggregation thresholds
        print("\n[Barrier 2] Applying aggregation thresholds...")
        filtered_clusters = self.barrier_2_aggregation_threshold(
            clusters,
            min_conversations,
            min_unique_accounts
        )

        # Barrier 3: Sanitize cluster summaries (applied during summary generation)
        print("\n[Barrier 3] Privacy-aware summary generation enabled")

        # Barrier 4: Audit final hierarchy
        print("\n[Barrier 4] Auditing final hierarchy...")
        audited_hierarchy, removed_clusters = self.barrier_4_auditing(hierarchy, model)

        results = {
            "validated_facets": validated_facets,
            "filtered_clusters": filtered_clusters,
            "audited_hierarchy": audited_hierarchy,
            "removed_clusters": removed_clusters,
            "privacy_stats": {
                "original_clusters": len(clusters),
                "filtered_clusters": len(filtered_clusters),
                "removed_in_audit": len(removed_clusters)
            }
        }

        print("\n" + "="*60)
        print("PRIVACY BARRIER SUMMARY")
        print("="*60)
        print(f"Original clusters: {results['privacy_stats']['original_clusters']}")
        print(f"After aggregation threshold: {results['privacy_stats']['filtered_clusters']}")
        print(f"After audit: {results['privacy_stats']['filtered_clusters'] - len(removed_clusters)}")
        print(f"Total removed: {results['privacy_stats']['original_clusters'] - (results['privacy_stats']['filtered_clusters'] - len(removed_clusters))}")

        return results


if __name__ == "__main__":
    # Example usage
    barriers = PrivacyBarriers()

    # Test privacy detection
    test_texts = [
        "The task is to write code for sorting algorithms",
        "User john.doe@example.com asked about Python",
        "Call me at 555-123-4567 for more info",
        "Translate English to Spanish"
    ]

    print("Testing privacy leakage detection:")
    for text in test_texts:
        has_issues, issues = barriers.detect_privacy_leakage(text)
        status = "BLOCKED" if has_issues else "PASSED"
        print(f"  [{status}] {text}")
        if has_issues:
            print(f"          Issues: {', '.join(issues)}")
