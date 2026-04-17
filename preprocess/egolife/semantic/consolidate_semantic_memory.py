#!/usr/bin/env python3
"""
Script to test the semantic consolidation functionality.
Loads semantic extraction results and applies semantic consolidation across timestamps.
"""

import json
import os
from typing import Dict, Any
from tqdm import tqdm

from worldmm.memory.semantic import SemanticConsolidation
from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_semantic_extraction_results(json_file: str) -> Dict[str, Any]:
    """Load semantic extraction results from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    """Main processing function."""
    # Configuration

    NUM_TO_NAME = {
        1: "A1_JAKE",
        2: "A2_ALICE",
        3: "A3_TASHA",
        4: "A4_LUCIA",
        5: "A5_KATRINA",
        6: "A6_SHURE"
    }

    index = 1
    model_name = os.getenv("TRANSLATION_MODEL", "openai/gpt-oss-120b")
    safe_model_name = model_name.replace("/", "_")
    semantic_results_file = f"output/metadata/semantic_memory/{NUM_TO_NAME[index]}/semantic_extraction_results_{safe_model_name}.json"
    output_dir = f"output/metadata/semantic_memory/{NUM_TO_NAME[index]}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(semantic_results_file):
        logger.error(f"Semantic extraction results file not found: {semantic_results_file}")
        logger.error("Please run the semantic extraction first using extract_semantic_triples.py")
        return
    
    semantic_results = load_semantic_extraction_results(semantic_results_file)
    logger.info(f"Loaded semantic extraction results with {len(semantic_results.get('semantic_triples', {}))} timestamps")
    
    # Count total semantic triples before consolidation
    total_triples_before = sum(
        len(triples) for triples in semantic_results.get('semantic_triples', {}).values()
    )
    logger.info(f"Total semantic triples before consolidation: {total_triples_before}")

    embedding_model_name = os.getenv("SEMANTIC_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    logger.info(f"Using semantic embedding model: {embedding_model_name}")
    embedding_model = EmbeddingModel(text_model_name=embedding_model_name)
    embedding_model.load_model(model_type="text")

    llm_model = LLMModel(model_name=model_name)

    semantic_consolidation = SemanticConsolidation(llm_model, embedding_model)

    # Process with batch semantic consolidation
    logger.info("Processing with Semantic Consolidation...")

    # Sort timestamps chronologically
    timestamps = sorted(semantic_results.get('semantic_triples', {}).keys())
    
    # Accumulated results across timestamps
    accumulated_semantic_triples = []
    accumulated_episodic_evidence = []
    
    # Growing results structure with timestamp as key
    timestamped_results = {}
    
    for i, timestamp in tqdm(enumerate(timestamps), total=len(timestamps)):
        logger.info(f"Processing timestamp {timestamp} ({i+1}/{len(timestamps)})")
        
        current_triples = semantic_results['semantic_triples'].get(timestamp, [])
        current_evidence = semantic_results['episodic_evidence'].get(timestamp, [])
        
        # Transform current evidence to "{timestamp}_{idx}" format
        transformed_current_evidence = []
        for evidence_list in current_evidence:
            transformed_list = [f"{timestamp}_{idx}" for idx in evidence_list]
            transformed_current_evidence.append(transformed_list)
        
        # Prepare existing results (from previous timestamps)
        existing_results = (accumulated_semantic_triples.copy(), accumulated_episodic_evidence.copy())
        
        # Prepare new results (current timestamp) - now with transformed evidence
        new_results = (current_triples, transformed_current_evidence)
        
        # Apply semantic consolidation (returns consolidated triples, evidence, and triples to remove)
        consolidated_triples, consolidated_evidence, triples_to_remove = semantic_consolidation.batch_semantic_consolidation(
            existing_results, new_results
        )
        
        logger.debug(f"  Removing {len(triples_to_remove)} existing triples")
        
        # Remove triples from accumulated state
        # Create a set of triples to remove for efficient lookup
        triples_to_remove_set = set()
        for triple, evidence in triples_to_remove:
            triples_to_remove_set.add((tuple(triple), tuple(evidence)))
        
        # Filter out removed triples from accumulated state
        new_accumulated_triples = []
        new_accumulated_evidence = []
        for acc_triple, acc_evidence in zip(accumulated_semantic_triples, accumulated_episodic_evidence):
            triple_key = (tuple(acc_triple), tuple(acc_evidence))
            if triple_key not in triples_to_remove_set:
                new_accumulated_triples.append(acc_triple)
                new_accumulated_evidence.append(acc_evidence)
        
        accumulated_semantic_triples = new_accumulated_triples
        accumulated_episodic_evidence = new_accumulated_evidence
        
        logger.debug(f"  Accumulated state: {len(accumulated_semantic_triples)} triples (after removal)")
        
        # Add new consolidated triples to accumulated state for next iteration
        accumulated_semantic_triples.extend(consolidated_triples)
        accumulated_episodic_evidence.extend(consolidated_evidence)
        
        logger.debug(f"  Final accumulated state: {len(accumulated_semantic_triples)} triples (after addition)")
        
        # Create final_results for current state
        final_results = {
            "consolidated_semantic_triples": accumulated_semantic_triples,
        }

        # Save with timestamp as key
        timestamped_results[timestamp] = final_results

        logger.debug(f"  Updated results up to timestamp {timestamp}")
    
    # Count total semantic triples after consolidation
    total_triples_after = len(accumulated_semantic_triples)
    logger.info(f"Total semantic triples after consolidation: {total_triples_after}")
    logger.info(f"Reduction in triples: {total_triples_before - total_triples_after} ({((total_triples_before - total_triples_after) / total_triples_before * 100):.1f}%)")
    
    # Final save of all timestamped results
    output_file = os.path.join(output_dir, f"semantic_consolidation_results_{safe_model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timestamped_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results have been saved to: {output_file}")


if __name__ == "__main__":
    main()
