#!/usr/bin/env python3
"""
Test script for query expansion diversity.
"""

import os
import logging
from deepsearch.models import SearchState
from deepsearch.agents import query_expansion_agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-query-expansion")

# Test queries
TEST_QUERIES = [
    "How does machine learning work?",
    "What are the best practices for cybersecurity?",
    "Explain quantum computing",
    "What are the effects of climate change?",
    "How to optimize Python code for performance?"
]

def calculate_similarity(query1, query2):
    """Calculate Jaccard similarity between two queries."""
    query1_words = set(query1.lower().split())
    query2_words = set(query2.lower().split())

    overlap = len(query1_words.intersection(query2_words))
    union = len(query1_words.union(query2_words))

    return overlap / union if union > 0 else 0

def test_query_expansion(query):
    """Test the query expansion for a specific query."""
    print(f"\n{'='*80}")
    print(f"TESTING QUERY: {query}")
    print(f"{'='*80}")

    # Initialize state
    state = SearchState(original_query=query)

    # Run query expansion
    state = query_expansion_agent(state)

    # Print generated queries
    print(f"\nGENERATED {len(state.generated_queries)} QUERIES:")
    for i, expanded_query in enumerate(state.generated_queries):
        print(f"{i+1}. {expanded_query}")

    # Analyze diversity
    print("\nQUERY SIMILARITIES (Jaccard index):")
    total_similarity = 0
    comparison_count = 0

    for i in range(len(state.generated_queries)):
        for j in range(i+1, len(state.generated_queries)):
            query1 = state.generated_queries[i]
            query2 = state.generated_queries[j]
            similarity = calculate_similarity(query1, query2)
            total_similarity += similarity
            comparison_count += 1
            print(f"Query {i+1} vs Query {j+1}: {similarity:.2f}")

    # Calculate average similarity (lower is better, indicates more diversity)
    if comparison_count > 0:
        avg_similarity = total_similarity / comparison_count
        print(f"\nAVERAGE SIMILARITY: {avg_similarity:.2f} (lower is better)")

    return state.generated_queries

def main():
    """Main test function."""
    all_results = {}

    for query in TEST_QUERIES:
        all_results[query] = test_query_expansion(query)

    print("\n\nSUMMARY OF ALL TEST QUERIES:")
    for query, results in all_results.items():
        print(f"\n{query}")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result}")

if __name__ == "__main__":
    main()