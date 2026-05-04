#!/usr/bin/env python3
"""
Script to process EgoLifeCap captions using Semantic Extraction functionality.
Groups caption data into chunks of `period`, retrieves corresponding OpenIE results (episodic triples) 
and extracts semantic knowledge from them.
"""

import argparse
import json
import logging
import os
from typing import List, Dict, Any

from worldmm.memory.semantic import SemanticExtraction
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.llm import LLMModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VALID_SUBJECTS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
period = 10

def load_caption_data(json_file: str) -> List[Dict]:
    """Load caption data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_openie_results(json_file: str) -> Dict[str, Any]:
    """Load OpenIE results from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def group_captions_and_get_openie_triples(caption_data: List[Dict], 
                                        openie_data: Dict[str, Any]) -> Dict[str, List[List[str]]]:
    """
    Group caption data into chunks of `period` and retrieve corresponding OpenIE triples.
    
    Args:
        caption_data: List of caption dictionaries with 'text', 'date', 'end_time' fields
        openie_data: Dictionary containing 'triple_results' from OpenIE processing
    
    Returns:
        Dictionary mapping group keys to lists of episodic triples
    """
    if 'triple_results' not in openie_data:
        raise ValueError("OpenIE data must contain 'triple_results' key")
    
    episodic_triples_batch = {}
    
    # Group captions into chunks of `period`
    for i in range(0, len(caption_data), period):
        chunk = caption_data[i:i+period]
        
        # Create key from the last item in the group: date[-1] + end_time.zfill(8)
        last_item = chunk[-1]
        group_key = last_item['date'][-1] + last_item['end_time'].zfill(8)
        
        # Collect all triples for this group from OpenIE results
        group_triples = []
        
        for caption_item in chunk:
            # Create hash to match with OpenIE results
            text_hash = compute_mdhash_id(caption_item['text'], prefix="chunk-")
            
            # Get triples for this caption - must exist in OpenIE results
            if text_hash not in openie_data['triple_results']:
                raise ValueError(f"Text hash {text_hash} not found in OpenIE results for text: {caption_item['text'][:100]}...")
            group_triples.extend(openie_data['triple_results'][text_hash])
        
        episodic_triples_batch[group_key] = group_triples
    
    return episodic_triples_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, choices=VALID_SUBJECTS,
                        help="EgoLife subject ID (e.g., A1_JAKE)")
    parser.add_argument("--caption-file", default=None,
                        help="Path to {subject}_30sec.json (default: data/EgoLife/EgoLifeCap/{subject}/{subject}_30sec.json)")
    parser.add_argument("--openie-file", default=None,
                        help="Path to openie_results_*.json (default derived from --subject)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: output/metadata/semantic_memory/{subject})")
    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_args()
    subject = args.subject
    model_name = os.getenv("TRANSLATION_MODEL", "openai/gpt-oss-120b")
    safe_model_name = model_name.replace("/", "_")

    caption_file = args.caption_file or f"data/EgoLife/EgoLifeCap/{subject}/{subject}_30sec.json"
    openie_results_file = args.openie_file or f"output/metadata/episodic_memory/{subject}/openie_results_{safe_model_name}.json"
    output_dir = args.output_dir or f"output/metadata/semantic_memory/{subject}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(caption_file):
        logger.error(f"Caption file not found: {caption_file}")
        return
    
    caption_data = load_caption_data(caption_file)
    logger.info(f"Loaded {len(caption_data)} caption entries")
    
    if not os.path.exists(openie_results_file):
        logger.error(f"OpenIE results file not found: {openie_results_file}")
        logger.error("Please run the OpenIE processing first using extract_episodic_triples.py")
        return
    
    openie_data = load_openie_results(openie_results_file)
    logger.info(f"Loaded OpenIE results with {len(openie_data.get('triple_results', {}))} chunks")
    
    # Group captions and extract corresponding episodic triples
    logger.info("Grouping captions into chunks of `period` and extracting corresponding episodic triples...")
    try:
        episodic_triples_batch = group_captions_and_get_openie_triples(caption_data, openie_data)
        logger.info(f"Created {len(episodic_triples_batch)} caption groups")
        
        # Print statistics
        total_triples = sum(len(triples) for triples in episodic_triples_batch.values())
        avg_triples = total_triples / len(episodic_triples_batch) if episodic_triples_batch else 0
        logger.info(f"Total episodic triples: {total_triples}")
        logger.info(f"Average triples per group: {avg_triples:.2f}")
        
    except Exception as e:
        logger.error(f"Error grouping captions and extracting episodic triples: {e}")
        return
    
    llm_model = LLMModel(model_name=model_name)

    semantic_extraction = SemanticExtraction(llm_model)

    # Process with batch semantic extraction
    print("Processing with Semantic Extraction...")
    semantic_triples_results, _ = semantic_extraction.batch_semantic_extraction(
        episodic_triples_batch, 
        output_dir=output_dir
    )
    
    total_semantic_triples = sum(len(result) for result in semantic_triples_results.values())
    print(f"Total semantic triples extracted: {total_semantic_triples}")
    
    print(f"Results have been saved to: {output_dir}/semantic_extraction_results_{safe_model_name}.json")

if __name__ == "__main__":
    main()
