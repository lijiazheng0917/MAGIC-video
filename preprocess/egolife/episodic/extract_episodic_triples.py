#!/usr/bin/env python3
"""
Script to process EgoLifeCap captions using OpenIE functionality.
Extracts Named Entities and Triples from the caption text, then reformats the results to timestamp-based structure.
"""

import argparse
import json
import logging
import os
from typing import List, Dict, Any

from worldmm.memory.episodic.openie import OpenIE
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.llm import LLMModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VALID_SUBJECTS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]


def load_caption_data(json_file: str) -> List[Dict]:
    """Load caption data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_text_passages(caption_data: List[Dict]) -> List[str]:
    """Extract text passages from caption data."""
    passages = []
    for item in caption_data:
        if 'text' in item:
            passages.append(item['text'])
    return passages


def create_episodic_triples_results(caption_data: List[Dict], 
                                  openie_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the episodic triples results dictionary.
    
    Args:
        caption_data: List of caption dictionaries with 'text', 'date', 'end_time', 'video_path' fields
        openie_data: Dictionary containing 'triple_results' from OpenIE processing
    
    Returns:
        Dictionary with 'episodic_triples' and 'raw_video' keys
    """
    if 'triple_results' not in openie_data:
        raise ValueError("OpenIE data must contain 'triple_results' key")
    
    episodic_triples = {}
    raw_video = {}
    
    # Process each caption individually (one-by-one)
    for caption_item in caption_data:
        # Create timestamp key from each item: date[-1] + end_time.zfill(8)
        timestamp = caption_item['date'][-1] + caption_item['end_time'].zfill(8)
        
        # Create hash to match with OpenIE results
        text_hash = compute_mdhash_id(caption_item['text'], prefix="chunk-")
        
        # Get triples for this caption - must exist in OpenIE results
        if text_hash not in openie_data['triple_results']:
            print(f"Warning: Text hash {text_hash} not found in OpenIE results for text: {caption_item['text'][:100]}...")
            # Store empty lists for missing entries
            episodic_triples[timestamp] = []
            raw_video[timestamp] = [caption_item['video_path']]
            continue
        
        # Store results for this individual caption
        episodic_triples[timestamp] = openie_data['triple_results'][text_hash]
        raw_video[timestamp] = caption_item['video_path']
    
    return {
        'episodic_triples': episodic_triples,
        'raw_video': raw_video
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, choices=VALID_SUBJECTS,
                        help="EgoLife subject ID (e.g., A1_JAKE)")
    parser.add_argument("--input-file", default=None,
                        help="Path to {subject}_30sec.json (default: data/EgoLife/EgoLifeCap/{subject}/{subject}_30sec.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: output/metadata/episodic_memory/{subject})")
    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_args()
    subject = args.subject
    input_file = args.input_file or f"data/EgoLife/EgoLifeCap/{subject}/{subject}_30sec.json"
    output_dir = args.output_dir or f"output/metadata/episodic_memory/{subject}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    caption_data = load_caption_data(input_file)
    logger.info(f"Loaded {len(caption_data)} caption entries")
    
    # Extract text passages
    text_passages = extract_text_passages(caption_data)
    logger.info(f"Extracted {len(text_passages)} text passages")
    
    llm_model = LLMModel(model_name=os.getenv("TRANSLATION_MODEL", "openai/gpt-oss-120b"))
    safe_model_name = llm_model.model_name.replace("/", "_")
    
    # Initialize OpenIE
    openie_processor = OpenIE(llm_model)
    
    # Process with batch OpenIE
    logger.info("Processing with OpenIE...")

    ner_results, triple_results = openie_processor.batch_openie(text_passages, output_dir=output_dir)
    
    total_entities = sum(len(result) for result in ner_results.values())
    total_triples = sum(len(result) for result in triple_results.values())
    logger.info(f"Total unique entities extracted: {total_entities}")
    logger.info(f"Total triples extracted: {total_triples}")

    logger.info(f"Successfully saved OpenIE results to: {output_dir}/openie_results_{safe_model_name}.json")
    
    # Load the OpenIE results that were just saved
    openie_results_file = os.path.join(output_dir, f"openie_results_{safe_model_name}.json")
    
    if not os.path.exists(openie_results_file):
        logger.error(f"OpenIE results file not found: {openie_results_file}")
        return
    
    with open(openie_results_file, 'r', encoding='utf-8') as f:
        openie_data = json.load(f)
    
    logger.info(f"Loaded OpenIE results with {len(openie_data.get('triple_results', {}))} text chunks")
    
    # Create episodic triples results
    results = create_episodic_triples_results(caption_data, openie_data)
    
    # Print statistics
    total_triples_reformat = sum(len(triples) for triples in results['episodic_triples'].values())
    avg_triples = total_triples_reformat / len(results['episodic_triples']) if results['episodic_triples'] else 0
    
    logger.info(f"Total episodic triples: {total_triples_reformat}")
    logger.info(f"Average triples per timestamp: {avg_triples:.2f}")
    
    # Save results
    output_file = os.path.join(output_dir, f"episodic_triple_results_{safe_model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully saved episodic triples results to: {output_file}")


if __name__ == "__main__":
    main()
