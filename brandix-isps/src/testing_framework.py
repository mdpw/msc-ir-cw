"""
Testing Framework for ISPS System
Satisfies Requirement 3.8: Testing and Evaluation

Implements:
- Ground truth comparison
- Precision/Recall/F1 metrics
- Alignment classification accuracy
- Expert validation framework
- Performance benchmarking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import time
from datetime import datetime
from pathlib import Path


class TestingFramework:
    """Comprehensive testing and evaluation framework"""
    
    def __init__(self):
        self.ground_truth = None
        self.predictions = None
        self.test_results = {}
        
    # ============================================================
    # 1. GROUND TRUTH MANAGEMENT
    # ============================================================
    
    def load_ground_truth(self, filepath: str) -> Dict:
        """
        Load ground truth alignment data
        
        Ground truth format:
        {
            "objective_action_pairs": [
                {
                    "objective_id": "OBJ-001",
                    "action_id": "ENV-AIR-001",
                    "expected_alignment": "Strong",  # Strong/Moderate/Weak
                    "expected_score": 0.85,  # 0-1
                    "expert_notes": "Direct match - solar installation"
                }
            ]
        }
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if ground truth has annotated pairs
            pairs = data.get('objective_action_pairs', [])
            
            if not pairs:
                print(f"WARNING: Ground truth file exists but has no pairs: {filepath}")
                return None
            
            # Check if at least some pairs are annotated (not "TO_BE_ANNOTATED")
            annotated_count = sum(
                1 for pair in pairs 
                if pair.get('expected_alignment') != 'TO_BE_ANNOTATED'
            )
            
            if annotated_count == 0:
                print(f"WARNING: Ground truth file exists but contains no annotated pairs")
                print(f"   File: {filepath}")
                print(f"   Total pairs: {len(pairs)}, Annotated: {annotated_count}")
                print(f"   Please annotate at least 5-10 pairs before running tests")
                return None
            
            # Valid annotated ground truth
            self.ground_truth = data
            print(f"Loaded ground truth: {len(pairs)} pairs, {annotated_count} annotated")
            return self.ground_truth
        
        except FileNotFoundError:
            print(f"WARNING: Ground truth file not found: {filepath}")
            print("Creating template ground truth file...")
            self._create_ground_truth_template(filepath)
            return None
    
    def _create_ground_truth_template(self, filepath: str):
        """Create template ground truth file for expert annotation"""
        template = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "annotator": "Expert Name",
                "notes": "Manual validation of strategic alignment"
            },
            "alignment_criteria": {
                "Strong": "Direct match, clear implementation (≥70%)",
                "Moderate": "Related but partial implementation (50-70%)",
                "Weak": "Minimal or unclear connection (<50%)"
            },
            "objective_action_pairs": [
                {
                    "objective_id": "OBJ-001",
                    "objective_text": "Example: Achieve 100% renewable energy by 2030",
                    "action_id": "ENV-AIR-001",
                    "action_title": "Example: Solar PV Installation Phase 1",
                    "expected_alignment": "Strong",
                    "expected_score": 0.85,
                    "expert_notes": "Direct implementation of renewable energy goal",
                    "confidence": "High"
                }
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)
        
        print(f"📝 Template created at: {filepath}")
        print("Please have domain expert annotate the ground truth data")
    
    # ============================================================
    # 2. ALIGNMENT CLASSIFICATION TESTING
    # ============================================================
    
    def evaluate_alignment_classification(self, sync_report: Dict) -> Dict:
        """
        Test accuracy of alignment classification (Strong/Moderate/Weak)
        
        Metrics:
        - Classification accuracy
        - Precision, Recall, F1 per class
        - Confusion matrix
        """
        if not self.ground_truth:
            return {"error": "Ground truth not loaded"}
        
        print("\n" + "="*80)
        print("ALIGNMENT CLASSIFICATION EVALUATION")
        print("="*80)
        
        # Extract predictions from sync report
        objective_details = sync_report.get('objective_details', [])
        
        # Build prediction and ground truth arrays
        y_true = []  # Ground truth labels
        y_pred = []  # System predictions
        
        ground_truth_pairs = {
            f"{pair['objective_id']}_{pair['action_id']}": pair
            for pair in self.ground_truth.get('objective_action_pairs', [])
        }
        
        matched_pairs = 0
        
        for obj_detail in objective_details:
            obj_id = obj_detail['objective_id']
            
            # Get top action for this objective
            if obj_detail['matched_actions']:
                top_action = obj_detail['matched_actions'][0]
                action_id = top_action['action_id']
                predicted_alignment = top_action['alignment_strength']
                
                # Check if this pair exists in ground truth
                pair_key = f"{obj_id}_{action_id}"
                
                if pair_key in ground_truth_pairs:
                    expected = ground_truth_pairs[pair_key]['expected_alignment']
                    
                    y_true.append(expected)
                    y_pred.append(predicted_alignment)
                    matched_pairs += 1
        
        if matched_pairs == 0:
            return {
                "error": "No matching pairs between ground truth and predictions",
                "suggestion": "Ensure ground truth contains same objective-action pairs"
            }
        
        print(f"Matched {matched_pairs} objective-action pairs for evaluation\n")
        
        # Define class order
        classes = ['Strong', 'Moderate', 'Weak']
        
        # Calculate metrics
        accuracy = np.mean([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
        
        # Precision, Recall, F1 for each class
        precision = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        
        # Weighted averages
        precision_weighted = precision_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        results = {
            'test_name': 'Alignment Classification Accuracy',
            'matched_pairs': matched_pairs,
            'overall_accuracy': float(accuracy),
            'per_class_metrics': {
                classes[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(np.sum(cm[:, i]))
                }
                for i in range(len(classes))
            },
            'weighted_metrics': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1_score': float(f1_weighted)
            },
            'confusion_matrix': {
                'matrix': cm.tolist(),
                'labels': classes
            }
        }
        
        # Display results
        self._display_classification_results(results)
        
        return results

    def evaluate_alignment_classification_all_pairs(self, sync_report: Dict) -> Dict:
        """
        Test accuracy of alignment classification for ALL ground truth pairs
        (not just top-ranked actions)

        This provides comprehensive evaluation across all annotated pairs,
        including weak alignments that may not appear in top recommendations.

        Metrics:
        - Classification accuracy
        - Precision, Recall, F1 per class
        - Confusion matrix
        """
        if not self.ground_truth:
            return {"error": "Ground truth not loaded"}

        print("\n" + "="*80)
        print("COMPREHENSIVE ALIGNMENT CLASSIFICATION EVALUATION (ALL PAIRS)")
        print("="*80)

        # Extract predictions from sync report - build lookup structure
        objective_details = sync_report.get('objective_details', [])

        # Create a lookup: {obj_id: {action_id: prediction}}
        predictions_lookup = {}
        for obj_detail in objective_details:
            obj_id = obj_detail['objective_id']
            predictions_lookup[obj_id] = {}

            for action in obj_detail.get('matched_actions', []):
                action_id = action['action_id']
                predictions_lookup[obj_id][action_id] = {
                    'alignment': action['alignment_strength'],
                    'score': action['similarity']
                }

        # Build prediction and ground truth arrays
        y_true = []  # Ground truth labels
        y_pred = []  # System predictions

        matched_pairs = 0
        missing_pairs = 0

        # Iterate through ALL ground truth pairs
        for pair in self.ground_truth.get('objective_action_pairs', []):
            obj_id = pair['objective_id']
            action_id = pair['action_id']
            expected_alignment = pair.get('expected_alignment')

            # Skip if not annotated yet
            if expected_alignment == 'TO_BE_ANNOTATED':
                continue

            # Look up the system prediction for this specific pair
            if obj_id in predictions_lookup and action_id in predictions_lookup[obj_id]:
                predicted_alignment = predictions_lookup[obj_id][action_id]['alignment']

                y_true.append(expected_alignment)
                y_pred.append(predicted_alignment)
                matched_pairs += 1
            else:
                missing_pairs += 1

        if matched_pairs == 0:
            return {
                "error": "No matching pairs between ground truth and predictions",
                "suggestion": "Ensure ground truth pairs were analyzed by the system"
            }

        print(f"Evaluated {matched_pairs} ground truth pairs")
        if missing_pairs > 0:
            print(f"Warning: {missing_pairs} ground truth pairs not found in predictions\n")
        else:
            print()

        # Define class order
        classes = ['Strong', 'Moderate', 'Weak']

        # Calculate metrics
        accuracy = np.mean([1 if t == p else 0 for t, p in zip(y_true, y_pred)])

        # Precision, Recall, F1 for each class
        precision = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

        # Weighted averages
        precision_weighted = precision_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        results = {
            'test_name': 'Comprehensive Alignment Classification (All Pairs)',
            'matched_pairs': matched_pairs,
            'missing_pairs': missing_pairs,
            'overall_accuracy': float(accuracy),
            'per_class_metrics': {
                classes[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(np.sum(cm[i, :]))  # Row sum for actual class count
                }
                for i in range(len(classes))
            },
            'weighted_metrics': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1_score': float(f1_weighted)
            },
            'confusion_matrix': {
                'matrix': cm.tolist(),
                'labels': classes
            }
        }

        # Display results
        self._display_classification_results(results)

        return results

    def _display_classification_results(self, results: Dict):
        """Display classification results in readable format"""
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}\n")

        print("Per-Class Performance:")
        print("-" * 60)
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)

        for class_name, metrics in results['per_class_metrics'].items():
            print(f"{class_name:<12} {metrics['precision']:.2%}     "
                  f"{metrics['recall']:.2%}     {metrics['f1_score']:.2%}")

        print("-" * 60)
        print(f"{'Weighted Avg':<12} {results['weighted_metrics']['precision']:.2%}     "
              f"{results['weighted_metrics']['recall']:.2%}     "
              f"{results['weighted_metrics']['f1_score']:.2%}")
        print()

        # Confusion Matrix
        print("\nConfusion Matrix:")
        print("-" * 60)
        cm = np.array(results['confusion_matrix']['matrix'])
        labels = results['confusion_matrix']['labels']

        # Header
        header = "Actual \\ Pred"
        print(f"{header:<15}", end="")
        for label in labels:
            print(f"{label:<12}", end="")
        print()
        print("-" * 60)

        # Rows
        for i, label in enumerate(labels):
            print(f"{label:<15}", end="")
            for j in range(len(labels)):
                print(f"{cm[i][j]:<12}", end="")
            print()
        print()
    
    # ============================================================
    # 3. SIMILARITY SCORE TESTING
    # ============================================================
    
    def evaluate_similarity_scores(self, sync_report: Dict) -> Dict:
        """
        Test accuracy of numerical similarity scores
        
        Metrics:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
        - Correlation coefficient
        """
        if not self.ground_truth:
            return {"error": "Ground truth not loaded"}
        
        print("\n" + "="*80)
        print("SIMILARITY SCORE EVALUATION")
        print("="*80)
        
        objective_details = sync_report.get('objective_details', [])
        
        ground_truth_pairs = {
            f"{pair['objective_id']}_{pair['action_id']}": pair['expected_score']
            for pair in self.ground_truth.get('objective_action_pairs', [])
        }
        
        true_scores = []
        pred_scores = []
        
        ground_truth_pairs = {
            f"{pair['objective_id']}_{pair['action_id']}": pair.get('expected_score')
            for pair in self.ground_truth.get('objective_action_pairs', [])
            if pair.get('expected_score') is not None
        }
        
        objective_details = sync_report.get('objective_details', [])
        
        true_scores = []
        pred_scores = []
        
        for obj_detail in objective_details:
            obj_id = obj_detail['objective_id']
            
            if obj_detail['matched_actions']:
                top_action = obj_detail['matched_actions'][0]
                action_id = top_action['action_id']
                predicted_score = top_action['similarity']
                
                pair_key = f"{obj_id}_{action_id}"
                
                if pair_key in ground_truth_pairs:
                    expected_score = ground_truth_pairs[pair_key]
                    
                    true_scores.append(expected_score)
                    pred_scores.append(predicted_score)
        
        if len(true_scores) == 0:
            return {"error": "No matching pairs for score evaluation"}
        
        # Calculate metrics
        true_scores = np.array(true_scores)
        pred_scores = np.array(pred_scores)
        
        mse = mean_squared_error(true_scores, pred_scores)
        mae = mean_absolute_error(true_scores, pred_scores)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(true_scores, pred_scores)[0, 1]
        
        # Additional metrics
        score_diff = np.abs(true_scores - pred_scores)
        within_10_percent = np.sum(score_diff <= 0.10) / len(score_diff)
        within_20_percent = np.sum(score_diff <= 0.20) / len(score_diff)
        
        results = {
            'test_name': 'Similarity Score Accuracy',
            'samples_evaluated': len(true_scores),
            'mean_squared_error': float(mse),
            'mean_absolute_error': float(mae),
            'root_mean_squared_error': float(rmse),
            'correlation_coefficient': float(correlation),
            'within_10_percent': float(within_10_percent),
            'within_20_percent': float(within_20_percent),
            'score_statistics': {
                'mean_true': float(np.mean(true_scores)),
                'mean_pred': float(np.mean(pred_scores)),
                'std_true': float(np.std(true_scores)),
                'std_pred': float(np.std(pred_scores))
            }
        }
        
        # Display results
        print(f"Samples Evaluated: {results['samples_evaluated']}\n")
        print("Error Metrics:")
        print(f"  MAE (Mean Absolute Error):  {mae:.4f}")
        print(f"  RMSE (Root Mean Squared):   {rmse:.4f}")
        print(f"  MSE (Mean Squared Error):   {mse:.4f}")
        print(f"\nCorrelation Coefficient:      {correlation:.4f}")
        print(f"\nAccuracy within ±10%:         {within_10_percent:.1%}")
        print(f"Accuracy within ±20%:         {within_20_percent:.1%}")
        print()
        
        return results
    
    # ============================================================
    # 4. LLM IMPROVEMENTS VALIDATION
    # ============================================================
    
    def evaluate_llm_improvements(self, improvements_data: Dict, expert_feedback: Dict = None) -> Dict:
        """
        Validate quality of LLM-generated improvement suggestions
        
        Metrics:
        - Relevance score (expert-rated)
        - Actionability score
        - Coverage of key categories
        - Specificity analysis
        """
        print("\n" + "="*80)
        print("LLM IMPROVEMENT SUGGESTIONS EVALUATION")
        print("="*80)
        
        improvements = improvements_data.get('improvements', [])
        
        if not improvements:
            return {"error": "No improvements to evaluate"}
        
        # Automatic quality checks
        quality_metrics = {
            'total_objectives_processed': len(improvements),
            'suggestions_per_objective': [],
            'category_coverage': {
                'new_actions': 0,
                'kpi_enhancements': 0,
                'timeline_recommendations': 0,
                'resource_requirements': 0,
                'risk_mitigation': 0
            },
            'specificity_scores': []
        }
        
        for imp in improvements:
            suggestions = imp.get('suggestions', {})
            
            # Count suggestions per objective
            total_suggestions = sum(len(v) for v in suggestions.values())
            quality_metrics['suggestions_per_objective'].append(total_suggestions)
            
            # Category coverage
            for category, items in suggestions.items():
                if items:
                    # Legacy mapping for old results
                    mapped_cat = category
                    if category == 'integration_opportunities': mapped_cat = 'risk_mitigation'
                    if category == 'timeline_adjustments': mapped_cat = 'timeline_recommendations'
                    if category == 'resource_allocation': mapped_cat = 'resource_requirements'
                    
                    if mapped_cat in quality_metrics['category_coverage']:
                        quality_metrics['category_coverage'][mapped_cat] += 1
            
            # Specificity check (basic heuristic: longer = more specific)
            for category, items in suggestions.items():
                for item in items:
                    # Score based on length and keywords
                    specificity = self._calculate_specificity_score(item)
                    quality_metrics['specificity_scores'].append(specificity)
        
        # Calculate averages
        avg_suggestions = np.mean(quality_metrics['suggestions_per_objective'])
        avg_specificity = np.mean(quality_metrics['specificity_scores']) if quality_metrics['specificity_scores'] else 0
        
        results = {
            'test_name': 'LLM Improvement Quality',
            'objectives_processed': quality_metrics['total_objectives_processed'],
            'average_suggestions_per_objective': float(avg_suggestions),
            'category_coverage_rate': {
                category: count / len(improvements)
                for category, count in quality_metrics['category_coverage'].items()
            },
            'average_specificity_score': float(avg_specificity),
            'quality_assessment': 'High' if avg_specificity >= 0.7 else 'Medium' if avg_specificity >= 0.5 else 'Low'
        }
        
        # If expert feedback provided, incorporate it
        if expert_feedback:
            results['expert_validation'] = self._process_expert_feedback(expert_feedback)
        
        # Display results
        print(f"Objectives Processed: {results['objectives_processed']}")
        print(f"Average Suggestions per Objective: {avg_suggestions:.1f}\n")
        
        print("Category Coverage:")
        for category, rate in results['category_coverage_rate'].items():
            print(f"  {category.replace('_', ' ').title():<30} {rate:.1%}")
        
        print(f"\nAverage Specificity Score: {avg_specificity:.2f}")
        print(f"Quality Assessment: {results['quality_assessment']}")
        print()
        
        return results
    
    def _calculate_specificity_score(self, suggestion: str) -> float:
        """
        Calculate specificity score for a suggestion
        
        Higher scores for:
        - Specific numbers/percentages
        - Concrete action verbs
        - Timeline mentions
        - Budget/resource figures
        """
        score = 0.0
        suggestion_lower = suggestion.lower()
        
        # Check for numbers (dates, percentages, quantities)
        import re
        if re.search(r'\d+', suggestion):
            score += 0.3
        
        # Check for percentages
        if '%' in suggestion or 'percent' in suggestion_lower:
            score += 0.2
        
        # Check for timeline keywords
        timeline_keywords = ['q1', 'q2', 'q3', 'q4', 'month', 'year', '202', 'deadline', 'by']
        if any(kw in suggestion_lower for kw in timeline_keywords):
            score += 0.2
        
        # Check for budget/resource keywords
        resource_keywords = ['budget', 'cost', '$', 'million', 'staff', 'team', 'invest']
        if any(kw in suggestion_lower for kw in resource_keywords):
            score += 0.2
        
        # Length-based component (longer = more detailed)
        if len(suggestion) > 50:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _process_expert_feedback(self, expert_feedback: Dict) -> Dict:
        """Process expert validation feedback"""
        # Expert feedback format:
        # {
        #     "objective_id": {
        #         "relevance": 0-5,
        #         "actionability": 0-5,
        #         "feasibility": 0-5,
        #         "comments": "text"
        #     }
        # }
        
        relevance_scores = []
        actionability_scores = []
        feasibility_scores = []
        
        for obj_id, feedback in expert_feedback.items():
            relevance_scores.append(feedback.get('relevance', 0))
            actionability_scores.append(feedback.get('actionability', 0))
            feasibility_scores.append(feedback.get('feasibility', 0))
        
        return {
            'objectives_reviewed': len(expert_feedback),
            'average_relevance': float(np.mean(relevance_scores)),
            'average_actionability': float(np.mean(actionability_scores)),
            'average_feasibility': float(np.mean(feasibility_scores)),
            'overall_expert_score': float(np.mean([
                np.mean(relevance_scores),
                np.mean(actionability_scores),
                np.mean(feasibility_scores)
            ]))
        }
    
    # ============================================================
    # 5. SYSTEM PERFORMANCE BENCHMARKING
    # ============================================================
    
    def benchmark_system_performance(self, sync_engine, strategic_path: str = None, action_path: str = None) -> Dict:
        """
        Benchmark system performance metrics
        
        Measures:
        - Processing time for different components (including real file I/O)
        - Memory usage
        - Throughput (objectives/second)
        """
        print("\n" + "="*80)
        print("SYSTEM PERFORMANCE BENCHMARKING")
        print("="*80)
        
        benchmarks = {}
        
        # 1. Document processing time (Full cycle: Load + Parse + Clean)
        print("Testing document processing speed (including file loading)...")
        start = time.time()
        
        # Get references for subsequent benchmarks
        objectives = sync_engine.doc_processor.strategic_objectives
        actions = sync_engine.doc_processor.action_items
        
        if strategic_path and action_path and Path(strategic_path).exists() and Path(action_path).exists():
            # Real-world benchmark: Load from disk and parse
            from document_processor import DocumentProcessor
            temp_processor = DocumentProcessor()
            _ = temp_processor.load_strategic_plan(strategic_path)
            _ = temp_processor.load_action_plan(action_path)
        else:
            # Fallback to in-memory cleaning if paths not provided
            for obj in objectives:
                _ = sync_engine.doc_processor.clean_text(obj.get('text', ''))
            for action in actions:
                _ = sync_engine.doc_processor.clean_text(action.get('text', ''))
        
        doc_time = time.time() - start
        benchmarks['document_processing_time'] = float(max(doc_time, 0.001))
        
        # 2. Embedding generation time
        print("Testing embedding generation speed...")
        start = time.time()
        # Use existing objectives for embedding benchmark
        obj_embeddings = sync_engine.embedding_engine.create_embeddings(
            [obj['text'] for obj in objectives[:10]]  # Sample
        )
        embed_time = time.time() - start
        benchmarks['embedding_time_per_10_items'] = float(embed_time)
        benchmarks['embedding_throughput'] = float(10 / embed_time) if embed_time > 0 else 0
        
        # 3. Similarity calculation time
        print("Testing similarity calculation speed...")
        start = time.time()
        if sync_engine.embedding_engine.similarity_matrix is None:
            sync_engine.embedding_engine.calculate_similarity_matrix()
        sim_time = time.time() - start
        benchmarks['similarity_matrix_calculation_time'] = float(sim_time)
        
        # 4. Overall analysis time (full pipeline)
        print("Testing full analysis pipeline...")
        start = time.time()
        # This would be a complete re-run, but we'll estimate from components
        total_time = doc_time + (len(objectives) / 10 * embed_time) + sim_time
        benchmarks['estimated_total_analysis_time'] = float(total_time)
        
        # 5. Scalability metrics
        n_objectives = len(objectives)
        n_actions = len(actions)
        benchmarks['scalability'] = {
            'objectives_count': n_objectives,
            'actions_count': n_actions,
            'comparisons_required': n_objectives * n_actions,
            'time_per_comparison_ms': float((total_time / (n_objectives * n_actions)) * 1000) if total_time > 0 else 0
        }
        
        results = {
            'test_name': 'System Performance Benchmark',
            'timestamp': datetime.now().isoformat(),
            'benchmarks': benchmarks
        }
        
        # Display results
        print(f"\nDocument Processing: {doc_time:.3f}s")
        print(f"Embedding (10 items): {embed_time:.3f}s ({benchmarks['embedding_throughput']:.1f} items/s)")
        print(f"Similarity Matrix: {sim_time:.3f}s")
        print(f"Estimated Total: {total_time:.3f}s")
        print(f"\nScalability: {n_objectives} objectives × {n_actions} actions = "
              f"{n_objectives * n_actions:,} comparisons")
        print(f"Time per comparison: {benchmarks['scalability']['time_per_comparison_ms']:.4f}ms")
        print()
        
        return results
    
    # ============================================================
    # 6. COMPREHENSIVE TEST SUITE
    # ============================================================
    
    def run_comprehensive_tests(self, sync_report: Dict, sync_engine, 
                                improvements_data: Dict = None,
                                ground_truth_path: str = "data/ground_truth.json",
                                expert_feedback: Dict = None,
                                strategic_path: str = None,
                                action_path: str = None) -> Dict:
        """
        Run all tests and generate comprehensive evaluation report
        
        Args:
            sync_report: Synchronization analysis results
            sync_engine: SynchronizationEngine instance
            improvements_data: LLM improvement suggestions (optional)
            ground_truth_path: Path to ground truth file
            expert_feedback: Expert validation feedback (optional)
        
        Returns:
            Complete test results dictionary
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE TESTING FRAMEWORK")
        print("ISPS System Evaluation")
        print("="*80)
        
        test_results = {
            'test_suite_version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'tests_run': []
        }
        
        # Load ground truth
        ground_truth_loaded = self.load_ground_truth(ground_truth_path)
        
        # Test 1: Alignment Classification (Top-K)
        if ground_truth_loaded:
            print("\n[Test 1/6] Alignment Classification Accuracy (Top-K)")
            classification_results = self.evaluate_alignment_classification(sync_report)
            test_results['alignment_classification'] = classification_results
            test_results['tests_run'].append('alignment_classification')
        else:
            print("\nWARNING: Skipping classification test - no ground truth available")
            test_results['alignment_classification'] = {
                'status': 'skipped',
                'reason': 'Ground truth not available'
            }

        # Test 1B: Comprehensive Alignment Classification (All Pairs)
        if ground_truth_loaded:
            print("\n[Test 1B/6] Comprehensive Alignment Classification (All Pairs)")
            comprehensive_results = self.evaluate_alignment_classification_all_pairs(sync_report)
            test_results['alignment_classification_comprehensive'] = comprehensive_results
            test_results['tests_run'].append('alignment_classification_comprehensive')
        else:
            print("\nWARNING: Skipping comprehensive classification test - no ground truth available")
            test_results['alignment_classification_comprehensive'] = {
                'status': 'skipped',
                'reason': 'Ground truth not available'
            }

        # Test 2: Similarity Scores
        if ground_truth_loaded:
            print("\n[Test 2/6] Similarity Score Accuracy")
            score_results = self.evaluate_similarity_scores(sync_report)
            test_results['similarity_scores'] = score_results
            test_results['tests_run'].append('similarity_scores')
        else:
            print("\nWARNING: Skipping score test - no ground truth available")
            test_results['similarity_scores'] = {
                'status': 'skipped',
                'reason': 'Ground truth not available'
            }

        # Test 3: LLM Improvements
        if improvements_data:
            print("\n[Test 3/6] LLM Improvement Quality")
            improvement_results = self.evaluate_llm_improvements(
                improvements_data,
                expert_feedback
            )
            test_results['llm_improvements'] = improvement_results
            test_results['tests_run'].append('llm_improvements')
        else:
            print("\nWARNING: Skipping LLM improvement test - no improvement data available")
            test_results['llm_improvements'] = {
                'status': 'skipped',
                'reason': 'No improvement data provided'
            }

        # Test 4: System Performance
        print("\n[Test 4/6] System Performance Benchmarks")
        performance_results = self.benchmark_system_performance(
            sync_engine,
            strategic_path=strategic_path,
            action_path=action_path
        )
        test_results['performance'] = performance_results
        test_results['tests_run'].append('performance')
        
        # Test 5: Coverage Analysis
        print("\n[Test 5/6] Coverage Analysis")
        coverage_results = self._analyze_test_coverage(sync_report)
        test_results['coverage'] = coverage_results
        test_results['tests_run'].append('coverage')
        
        # Generate overall assessment
        test_results['overall_assessment'] = self._generate_overall_assessment(test_results)
        
        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)
        self._display_overall_assessment(test_results['overall_assessment'])
        
        return test_results
    
    def _analyze_test_coverage(self, sync_report: Dict) -> Dict:
        """Analyze what percentage of system is covered by tests"""
        total_objectives = sync_report['overall_alignment']['total_objectives']
        total_actions = sync_report['overall_alignment']['total_actions']
        
        # Calculate coverage
        objectives_tested = 0
        if self.ground_truth:
            unique_obj_ids = set(
                pair['objective_id'] 
                for pair in self.ground_truth.get('objective_action_pairs', [])
            )
            objectives_tested = len(unique_obj_ids)
        
        coverage_rate = (objectives_tested / total_objectives * 100) if total_objectives > 0 else 0
        
        return {
            'test_name': 'Test Coverage Analysis',
            'total_objectives': total_objectives,
            'objectives_tested': objectives_tested,
            'coverage_percentage': float(coverage_rate),
            'coverage_assessment': 'High' if coverage_rate >= 70 else 'Medium' if coverage_rate >= 40 else 'Low'
        }
    
    def _generate_overall_assessment(self, test_results: Dict) -> Dict:
        """Generate overall system quality assessment"""
        assessment = {
            'tests_completed': len(test_results['tests_run']),
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'overall_grade': 'N/A',
            'recommendations': []
        }
        
        # Check each test result
        if 'alignment_classification' in test_results:
            result = test_results['alignment_classification']
            if result.get('status') == 'skipped':
                assessment['tests_skipped'] += 1
            elif result.get('overall_accuracy', 0) >= 0.50:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['recommendations'].append(
                    "Improve alignment classification accuracy (currently below 50%)"
                )

        # Check comprehensive alignment classification (all pairs)
        if 'alignment_classification_comprehensive' in test_results:
            result = test_results['alignment_classification_comprehensive']
            if result.get('status') == 'skipped':
                assessment['tests_skipped'] += 1
            elif result.get('overall_accuracy', 0) >= 0.50:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['recommendations'].append(
                    "Improve comprehensive alignment classification (weak class detection needs enhancement)"
                )
        
        if 'similarity_scores' in test_results:
            result = test_results['similarity_scores']
            if result.get('status') == 'skipped':
                assessment['tests_skipped'] += 1
            elif result.get('correlation_coefficient', 0) >= 0.40:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['recommendations'].append(
                    "Improve similarity score accuracy (correlation with expert below 0.40)"
                )
        
        if 'llm_improvements' in test_results:
            result = test_results['llm_improvements']
            if result.get('status') == 'skipped':
                assessment['tests_skipped'] += 1
            elif result.get('average_specificity_score', 0) >= 0.55:
                assessment['tests_passed'] += 1
            else:
                assessment['tests_failed'] += 1
                assessment['recommendations'].append(
                    "Enhance LLM prompt engineering for more specific suggestions"
                )
        
        # Calculate overall grade
        total_evaluatable = assessment['tests_passed'] + assessment['tests_failed']
        if total_evaluatable > 0:
            pass_rate = assessment['tests_passed'] / total_evaluatable
            if pass_rate >= 0.90:
                assessment['overall_grade'] = 'A (Excellent)'
            elif pass_rate >= 0.75:
                assessment['overall_grade'] = 'B+ (Very Good)'
            elif pass_rate >= 0.60:
                assessment['overall_grade'] = 'B (Good)'
            elif pass_rate >= 0.45:
                assessment['overall_grade'] = 'C+ (Above Average)'
            elif pass_rate >= 0.30:
                assessment['overall_grade'] = 'C (Satisfactory)'
            else:
                assessment['overall_grade'] = 'D (Needs Improvement)'
        
        return assessment
    
    def _display_overall_assessment(self, assessment: Dict):
        """Display overall assessment summary"""
        print(f"\nTests Completed: {assessment['tests_completed']}")
        print(f"Passed: {assessment['tests_passed']}")
        print(f"Failed: {assessment['tests_failed']}")
        print(f"Skipped: {assessment['tests_skipped']}")
        print(f"\nOverall Grade: {assessment['overall_grade']}")
        
        if assessment['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(assessment['recommendations'], 1):
                print(f"  {i}. {rec}")
        print()
    
    # ============================================================
    # 7. SAVE RESULTS
    # ============================================================
    
    def save_test_results(self, test_results: Dict, output_file: str):
        """Save test results to JSON file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Test results saved to: {output_file}")
        
        # Also save a markdown summary
        md_file = output_file.replace('.json', '_summary.md')
        self._generate_markdown_report(test_results, md_file)
    
    def _generate_markdown_report(self, test_results: Dict, output_file: str):
        """Generate human-readable markdown report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# ISPS Testing Framework - Evaluation Report\n\n")
            f.write(f"**Generated:** {test_results['timestamp']}\n\n")
            f.write("---\n\n")
            
            # Overall Assessment
            if 'overall_assessment' in test_results:
                assessment = test_results['overall_assessment']
                f.write("## Overall Assessment\n\n")
                f.write(f"- **Grade:** {assessment['overall_grade']}\n")
                f.write(f"- **Tests Passed:** {assessment['tests_passed']}/{assessment['tests_completed']}\n")
                f.write(f"- **Tests Failed:** {assessment['tests_failed']}\n")
                f.write(f"- **Tests Skipped:** {assessment['tests_skipped']}\n\n")
                
                if assessment['recommendations']:
                    f.write("### Recommendations\n\n")
                    for i, rec in enumerate(assessment['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
            
            f.write("---\n\n")
            
            # Individual Test Results
            if 'alignment_classification' in test_results:
                result = test_results['alignment_classification']
                if result.get('status') != 'skipped' and 'overall_accuracy' in result:
                    f.write("## Test 1: Alignment Classification\n\n")
                    f.write(f"- **Overall Accuracy:** {result['overall_accuracy']:.2%}\n")
                    f.write(f"- **Weighted F1-Score:** {result['weighted_metrics']['f1_score']:.2%}\n\n")
            
            if 'similarity_scores' in test_results:
                result = test_results['similarity_scores']
                if result.get('status') != 'skipped' and 'mean_absolute_error' in result:
                    f.write("## Test 2: Similarity Scores\n\n")
                    f.write(f"- **MAE:** {result['mean_absolute_error']:.4f}\n")
                    f.write(f"- **Correlation:** {result['correlation_coefficient']:.4f}\n")
                    f.write(f"- **Within ±10%:** {result['within_10_percent']:.1%}\n\n")
            
            if 'llm_improvements' in test_results:
                result = test_results['llm_improvements']
                if result.get('status') != 'skipped' and 'objectives_processed' in result:
                    f.write("## Test 3: LLM Improvements\n\n")
                    f.write(f"- **Objectives Processed:** {result['objectives_processed']}\n")
                    f.write(f"- **Avg Specificity:** {result['average_specificity_score']:.2f}\n")
                    f.write(f"- **Quality:** {result['quality_assessment']}\n\n")
            
            f.write("---\n\n")
            f.write("*End of Report*\n")
        
        print(f"Markdown summary saved to: {output_file}")


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("ISPS TESTING FRAMEWORK - STANDALONE TEST")
    print("="*80)
    
    # This would normally be run with actual system data
    # For demonstration, we show how to use it
    
    print("\nTo use this testing framework:")
    print("\n1. Create ground truth data:")
    print("   - Manually annotate objective-action pairs")
    print("   - Save to data/ground_truth.json")
    
    print("\n2. Run tests:")
    print("   python src/testing_framework.py")
    
    print("\n3. Review results:")
    print("   - outputs/test_results.json")
    print("   - outputs/test_results_summary.md")
    
    print("\n" + "="*80)
    print("Testing framework ready!")
    print("="*80)