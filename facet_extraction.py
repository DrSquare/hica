"""
Facet Extraction Module for Clio Privacy-Preserving Classification
Based on arxiv.org/abs/2412.13678
"""

import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class FacetExtractor:
    """
    Extracts facets from conversations while preserving privacy.
    Implements the first privacy layer from the Clio paper.
    """

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Facet definitions from Clio paper and OpenClio implementation
        self.facets = {
            "request": {
                "question": "What is the user's overall request for the assistant?",
                "prefill": "The user's overall request for the assistant is to",
                "privacy_instruction": "Omit any personally identifiable information, private data, or specific identifying details."
            },
            "task": {
                "question": "What task is the model being asked to perform?",
                "prefill": "The task is to",
                "privacy_instruction": "Focus on the general task type without including private information, personal details, or sensitive content."
            },
            "language": {
                "question": "What languages are present in this conversation? Provide full names for human languages (e.g., 'English', 'Spanish'), not language codes. Exclude language families.",
                "prefill": "The languages present are:",
                "privacy_instruction": "Only identify languages, not the content itself."
            },
            "concern_level": {
                "question": """Rate the safety concern level of this conversation on a 1-5 scale:
1 - Not concerning at all. Completely benign conversation.
2 - Slightly concerning. Minor issues but generally acceptable.
3 - Moderately concerning. Some problematic elements.
4 - Very concerning. Significant safety or policy issues.
5 - Extremely concerning. Immediate review needed.

Provide only the number.""",
                "prefill": "Concern level:",
                "privacy_instruction": "Assess concern level without revealing private conversation details."
            }
        }

    def extract_facet(self, conversation: str, facet_name: str, model: str = "gpt-4") -> str:
        """
        Extract a single facet from a conversation.

        Args:
            conversation: The conversation text to analyze
            facet_name: Name of the facet to extract (request, task, language, concern_level)
            model: OpenAI model to use

        Returns:
            Extracted facet value as a string
        """
        if facet_name not in self.facets:
            raise ValueError(f"Unknown facet: {facet_name}. Must be one of {list(self.facets.keys())}")

        facet_config = self.facets[facet_name]

        system_prompt = f"""You are analyzing conversations to extract specific attributes while strictly preserving privacy.

CRITICAL PRIVACY INSTRUCTION: {facet_config['privacy_instruction']}

Do not include:
- Names, usernames, or personal identifiers
- Email addresses, phone numbers, or contact information
- Addresses or specific locations beyond country/region level
- Company names or proprietary information
- Any other personally identifiable or sensitive information

Extract only the requested attribute in a general, anonymized form."""

        user_prompt = f"""Analyze the following conversation and answer this question:

{facet_config['question']}

Conversation:
{conversation}

{facet_config['prefill']}"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            result = response.choices[0].message.content.strip()

            # For prefilled responses, combine prefill with result
            if facet_config['prefill'] and not result.startswith(facet_config['prefill']):
                result = facet_config['prefill'] + " " + result

            return result

        except Exception as e:
            print(f"Error extracting facet '{facet_name}': {str(e)}")
            return f"{facet_config['prefill']} [extraction_error]"

    def extract_all_facets(self, conversation: str, model: str = "gpt-4") -> Dict[str, str]:
        """
        Extract all facets from a conversation.

        Args:
            conversation: The conversation text to analyze
            model: OpenAI model to use

        Returns:
            Dictionary mapping facet names to extracted values
        """
        facets = {}
        for facet_name in self.facets.keys():
            facets[facet_name] = self.extract_facet(conversation, facet_name, model)

        return facets

    def batch_extract_facets(self, conversations: List[str],
                            facet_names: List[str] = None,
                            model: str = "gpt-4") -> List[Dict[str, str]]:
        """
        Extract facets from multiple conversations.

        Args:
            conversations: List of conversation texts
            facet_names: List of facet names to extract (None = all facets)
            model: OpenAI model to use

        Returns:
            List of dictionaries containing extracted facets for each conversation
        """
        if facet_names is None:
            facet_names = list(self.facets.keys())

        results = []
        for i, conversation in enumerate(conversations):
            print(f"Processing conversation {i+1}/{len(conversations)}...")
            facets = {}
            for facet_name in facet_names:
                facets[facet_name] = self.extract_facet(conversation, facet_name, model)
            results.append(facets)

        return results


if __name__ == "__main__":
    # Example usage
    extractor = FacetExtractor()

    # Sample conversation from the paper's domain
    sample_conversation = """
    User: Can you help me write a Python function to sort a list of numbers?

    Assistant: Of course! Here's a simple function to sort a list:

    def sort_numbers(numbers):
        return sorted(numbers)

    User: Thanks! Can you also show me how to do it in descending order?

    Assistant: Sure! Just add the reverse parameter:

    def sort_numbers_desc(numbers):
        return sorted(numbers, reverse=True)
    """

    print("Extracting facets from sample conversation...")
    facets = extractor.extract_all_facets(sample_conversation)

    print("\nExtracted Facets:")
    for facet_name, value in facets.items():
        print(f"\n{facet_name.upper()}:")
        print(f"  {value}")
