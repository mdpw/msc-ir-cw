"""
Test Runner Script
Runs comprehensive tests on ISPS system and generates evaluation report
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from testing_framework import TestingFramework
from synchronization_engine import SynchronizationEngine
from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from vector_store import VectorStore


def run_tests_for_year(year: str = "2026"):
    """
    Run comprehensive tests for a specific year
    
    Args:
        year: Year to test (2026-2030)
    """
    print("="*80)
    print(f"RUNNING COMPREHENSIVE TESTS FOR YEAR {year}")
    print("="*80)
    
    # Paths
    outputs_dir = Path("outputs") / year
    sync_report_path = outputs_dir / "synchronization_report.json"
    improvements_path = outputs_dir / "improvements.json"
    ground_truth_path = Path("data") / "ground_truth" / f"{year}_ground_truth.json"
    test_output_path = outputs_dir / "test_results.json"
    
    # Check if synchronization report exists
    if not sync_report_path.exists():
        print(f"\n❌ Error: Synchronization report not found for {year}")
        print(f"Expected at: {sync_report_path}")
        print("\n👉 Please run analysis first:")
        print(f"   1. Go to 'Run Analysis' page")
        print(f"   2. Select year {year}")
        print(f"   3. Complete analysis")
        return None
    
    # Load synchronization report
    print(f"\n📊 Loading synchronization report...")
    with open(sync_report_path, 'r', encoding='utf-8') as f:
        sync_report = json.load(f)
    print(f"✅ Loaded report with {sync_report['overall_alignment']['total_objectives']} objectives")
    
    # Load improvements if available
    improvements_data = None
    if improvements_path.exists():
        print(f"\n💡 Loading improvement suggestions...")
        with open(improvements_path, 'r', encoding='utf-8') as f:
            improvements_data = json.load(f)
        print(f"✅ Loaded {len(improvements_data.get('improvements', []))} improvement sets")
    else:
        print(f"\n⚠️ No improvements file found - skipping LLM evaluation")
    
    # Initialize synchronization engine for benchmarking
    print(f"\n⚙️ Initializing system components for benchmarking...")
    
    # Load documents
    upload_dir = Path("data/uploaded") / year
    strategic_path = upload_dir / "strategic_plan.docx"
    action_path = upload_dir / "action_plan.docx"
    
    processor = DocumentProcessor()
    objectives = processor.load_strategic_plan(str(strategic_path))
    actions = processor.load_action_plan(str(action_path))
    
    # Create engines
    engine = EmbeddingEngine()
    vs = VectorStore(dimension=384)
    
    # Create sync engine
    sync_engine = SynchronizationEngine(processor, engine, vs)
    
    # Embed for benchmarking
    engine.embed_objectives(objectives)
    engine.embed_actions(actions)
    
    print(f"✅ System components initialized")
    
    # Initialize testing framework
    print(f"\n🧪 Initializing testing framework...")
    testing_framework = TestingFramework()
    
    # Run comprehensive tests
    test_results = testing_framework.run_comprehensive_tests(
        sync_report=sync_report,
        sync_engine=sync_engine,
        improvements_data=improvements_data,
        ground_truth_path=str(ground_truth_path),
        expert_feedback=None  # Can be added later
    )
    
    # Save results
    testing_framework.save_test_results(test_results, str(test_output_path))
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - {test_output_path}")
    print(f"  - {test_output_path.with_suffix('')}_summary.md")
    
    return test_results


def create_sample_ground_truth(year: str = "2026"):
    """
    Create sample ground truth file with instructions for manual annotation
    
    Args:
        year: Year to create ground truth for
    """
    print("="*80)
    print(f"CREATING SAMPLE GROUND TRUTH FOR YEAR {year}")
    print("="*80)
    
    # Load synchronization report to get actual objectives and actions
    outputs_dir = Path("outputs") / year
    sync_report_path = outputs_dir / "synchronization_report.json"
    
    if not sync_report_path.exists():
        print(f"\n❌ Error: No synchronization report found for {year}")
        print("Please run analysis first.")
        return
    
    with open(sync_report_path, 'r', encoding='utf-8') as f:
        sync_report = json.load(f)
    
    # Create ground truth directory
    gt_dir = Path("data/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    gt_path = gt_dir / f"{year}_ground_truth.json"
    
    # Get top 10 objectives for annotation
    objective_details = sync_report['objective_details'][:10]
    
    # Build template
    template = {
        "metadata": {
            "year": year,
            "created": "2026-01-28",
            "annotator": "Domain Expert Name",
            "instructions": "Please review each objective-action pair and provide your expert assessment",
            "notes": "This is a sample with top 10 objectives. Expand as needed."
        },
        "alignment_criteria": {
            "Strong": "Direct match, clear implementation of objective (≥70% alignment)",
            "Moderate": "Related but partial implementation (50-70% alignment)",
            "Weak": "Minimal or unclear connection (<50% alignment)"
        },
        "scoring_guide": {
            "1.0": "Perfect alignment - action directly achieves objective",
            "0.8-0.9": "Strong alignment - action substantially addresses objective",
            "0.6-0.7": "Good alignment - action clearly supports objective",
            "0.5-0.6": "Moderate alignment - action partially supports objective",
            "0.3-0.5": "Weak alignment - action has minor connection",
            "0.0-0.3": "Very weak alignment - action barely relates"
        },
        "objective_action_pairs": []
    }
    
    # Add top matches for each objective
    for obj_detail in objective_details:
        obj_id = obj_detail['objective_id']
        obj_text = obj_detail['objective']
        
        # Get top 3 actions
        for action in obj_detail['matched_actions'][:3]:
            pair = {
                "objective_id": obj_id,
                "objective_text": obj_text,
                "objective_pillar": obj_detail.get('pillar', 'Unknown'),
                "action_id": action['action_id'],
                "action_title": action['title'],
                "action_pillar": action.get('pillar', 'Unknown'),
                "system_prediction": {
                    "alignment": action['alignment_strength'],
                    "score": action['similarity']
                },
                "expected_alignment": "TO_BE_ANNOTATED",  # Expert fills this
                "expected_score": 0.0,  # Expert fills this
                "expert_notes": "",  # Expert adds notes
                "confidence": "TO_BE_ANNOTATED"  # High/Medium/Low
            }
            
            template['objective_action_pairs'].append(pair)
    
    # Save template
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✅ Sample ground truth template created!")
    print(f"📝 File: {gt_path}")
    print(f"📊 Contains: {len(template['objective_action_pairs'])} pairs to annotate")
    
    print("\n" + "="*80)
    print("NEXT STEPS FOR EXPERT ANNOTATION")
    print("="*80)
    print("\n1. Open the ground truth file:")
    print(f"   {gt_path}")
    
    print("\n2. For each pair, replace:")
    print("   - 'TO_BE_ANNOTATED' with actual values")
    print("   - Add expert_notes with reasoning")
    
    print("\n3. Example annotation:")
    print("""   {
     "expected_alignment": "Strong",
     "expected_score": 0.85,
     "expert_notes": "Action directly implements renewable energy target",
     "confidence": "High"
   }""")
    
    print("\n4. Save the file and run tests again")
    print("\n" + "="*80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ISPS Testing Framework Runner")
    parser.add_argument(
        '--year',
        type=str,
        default='2026',
        choices=['2026', '2027', '2028', '2029', '2030'],
        help='Year to test (default: 2026)'
    )
    parser.add_argument(
        '--create-ground-truth',
        action='store_true',
        help='Create sample ground truth template instead of running tests'
    )
    
    args = parser.parse_args()
    
    if args.create_ground_truth:
        create_sample_ground_truth(args.year)
    else:
        run_tests_for_year(args.year)


if __name__ == "__main__":
    main()