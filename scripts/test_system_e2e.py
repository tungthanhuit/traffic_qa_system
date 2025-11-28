"""
End-to-End System Test Script

This script tests the entire QA system pipeline:
1. Takes a question from violations_test.json
2. Runs it through the complete retrieval + generation pipeline
3. Compares the retrieved violation_id with the expected violation_id
4. Stores all results with detailed metrics

Metrics tracked:
- Exact Match (violation_id matches)
- Retrieval Success (expected violation in top-k candidates)
- Response Quality
- System Performance
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.presentation.di_container import Container
from src.domain.models import QueryResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemE2ETester:
    """End-to-end system tester"""
    
    def __init__(self, container: Container, log_file: Optional[str] = None):
        """Initialize with dependency injection container"""
        self.container = container
        self.ask_question_use_case = container.ask_question_use_case()
        
        # Access internal components for detailed analysis
        self.vector_store = container.vs_adapter()
        self.kg = container.kg_adapter()
        self.embedding_service = container.embed_adapter()
        self.llm_service = container.llm_adapter()
        
        # Setup log file
        self.log_file = log_file
        if self.log_file:
            # Create/clear log file and write header
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("END-TO-END SYSTEM TEST LOG\n")
                f.write(f"Started at: {datetime.now().isoformat()}\n")
                f.write("="*80 + "\n\n")
        
    def log_to_file(self, message: str):
        """Write message to log file in real time"""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + "\n")
    
    def get_retrieval_candidates(
        self, 
        question: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Get retrieval candidates to check if expected violation is retrieved
        
        Returns:
            List of (violation_id, similarity_score, metadata) tuples
        """
        try:
            # Parse query to get action
            parsed = self.llm_service.parse_query(question)
            action = parsed.action if parsed.action else question
            
            # Get embedding
            query_vector = self.embedding_service.embed_text(action)
            
            # Get candidates (with lower threshold to see all results)
            candidates = self.vector_store.search_similar(
                query_embedding=query_vector,
                k=top_k,
                min_similarity=0.5,  # Lower threshold to see all candidates
                filter_metadata=None
            )
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting retrieval candidates: {e}")
            return []
    
    def test_single_question(
        self, 
        test_case: Dict,
        track_retrieval: bool = True
    ) -> Dict:
        """
        Test a single question through the entire system
        
        Args:
            test_case: Dict with test_id, question, violation_id
            track_retrieval: Whether to track retrieval candidates
            
        Returns:
            Dict with test results and metrics
        """
        test_id = test_case.get('test_id')
        question = test_case.get('question', '')
        expected_violation_id = test_case.get('violation_id')
        
        log_msg = f"\n{'='*80}\n"
        log_msg += f"Testing {test_id}\n"
        log_msg += f"Question: {question}\n"
        log_msg += f"Expected Violation ID: {expected_violation_id}\n"
        
        logger.info(log_msg)
        self.log_to_file(log_msg)
        
        start_time = time.time()
        
        try:
            # Get retrieval candidates first (for analysis)
            retrieval_candidates = []
            expected_rank = -1
            expected_similarity = 0.0
            
            if track_retrieval:
                retrieval_candidates = self.get_retrieval_candidates(question, top_k=10)
                
                # Log all retrieval candidates
                candidates_log = f"\n--- RETRIEVAL CANDIDATES ---\n"
                candidates_log += f"Total candidates found: {len(retrieval_candidates)}\n"
                for idx, (vid, similarity, metadata) in enumerate(retrieval_candidates):
                    candidates_log += f"  Rank {idx + 1}: {vid} (similarity: {similarity:.4f})\n"
                    if metadata:
                        candidates_log += f"    Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
                
                logger.info(candidates_log)
                self.log_to_file(candidates_log)
                
                # Check if expected violation is in candidates
                for idx, (vid, similarity, metadata) in enumerate(retrieval_candidates):
                    if vid == expected_violation_id:
                        expected_rank = idx + 1
                        expected_similarity = similarity
                        break
                
                rank_msg = f"\nRetrieval: Found {len(retrieval_candidates)} candidates\n"
                if expected_rank > 0:
                    rank_msg += f"✓ Expected violation at rank {expected_rank} with similarity {expected_similarity:.4f}\n"
                else:
                    rank_msg += f"✗ Expected violation NOT in top-{len(retrieval_candidates)} candidates!\n"
                
                logger.info(rank_msg)
                self.log_to_file(rank_msg)
            
            # Run through complete system
            response: QueryResponse = self.ask_question_use_case.execute(question)
            
            elapsed_time = time.time() - start_time
            
            # Extract retrieved violation ID
            retrieved_violation_id = None
            if response.violation_found:
                retrieved_violation_id = response.violation_found.id
            
            # Check if match
            exact_match = retrieved_violation_id == expected_violation_id
            
            result_msg = f"\n--- SYSTEM RESPONSE ---\n"
            result_msg += f"Retrieved Violation ID: {retrieved_violation_id}\n"
            result_msg += f"Match: {'✓ CORRECT' if exact_match else '✗ INCORRECT'}\n"
            result_msg += f"Response Time: {elapsed_time:.2f}s\n"
            result_msg += f"Answer: {response.answer}\n"
            if response.citation:
                result_msg += f"Citation: {response.citation}\n"
            if response.fine:
                result_msg += f"Fine: {response.fine.min_amount:,} - {response.fine.max_amount:,} VND\n"
            if response.legal_basis:
                result_msg += f"Legal Basis: {response.legal_basis.decree} - Article {response.legal_basis.article}"
                if response.legal_basis.clause:
                    result_msg += f", Clause {response.legal_basis.clause}"
                if response.legal_basis.point:
                    result_msg += f", Point {response.legal_basis.point}"
                result_msg += "\n"
            
            logger.info(result_msg)
            self.log_to_file(result_msg)
            
            # Build result
            result = {
                'test_id': test_id,
                'question': question,
                'expected_violation_id': expected_violation_id,
                'retrieved_violation_id': retrieved_violation_id,
                'exact_match': exact_match,
                'response_time_seconds': round(elapsed_time, 2),
                'answer': response.answer,
                'citation': response.citation,
                'fine': None,
                'legal_basis': None,
                'retrieval_metrics': {
                    'expected_in_candidates': expected_rank > 0,
                    'expected_rank': expected_rank if expected_rank > 0 else None,
                    'expected_similarity': round(expected_similarity, 4) if expected_rank > 0 else None,
                    'total_candidates': len(retrieval_candidates),
                    'all_candidates': [
                        {
                            'violation_id': vid,
                            'similarity': round(sim, 4),
                            'metadata': metadata
                        }
                        for vid, sim, metadata in retrieval_candidates
                    ]
                }
            }
            
            # Add fine info if available
            if response.fine:
                result['fine'] = {
                    'min_vnd': response.fine.min_amount,
                    'max_vnd': response.fine.max_amount
                }
            
            # Add legal basis if available
            if response.legal_basis:
                result['legal_basis'] = {
                    'decree': response.legal_basis.decree,
                    'article': response.legal_basis.article,
                    'clause': response.legal_basis.clause,
                    'point': response.legal_basis.point
                }
            
            return result
            
        except Exception as e:
            error_msg = f"\n✗ ERROR testing {test_id}: {e}\n"
            logger.error(error_msg, exc_info=True)
            self.log_to_file(error_msg)
            
            elapsed_time = time.time() - start_time
            
            return {
                'test_id': test_id,
                'question': question,
                'expected_violation_id': expected_violation_id,
                'retrieved_violation_id': None,
                'exact_match': False,
                'response_time_seconds': round(elapsed_time, 2),
                'error': str(e),
                'answer': None,
                'citation': None,
                'retrieval_metrics': {
                    'expected_in_candidates': False,
                    'expected_rank': None,
                    'expected_similarity': None,
                    'total_candidates': 0,
                    'all_candidates': []
                }
            }
    
    def run_all_tests(
        self, 
        test_file_path: str,
        max_tests: Optional[int] = None
    ) -> Dict:
        """
        Run all tests and generate comprehensive report
        
        Args:
            test_file_path: Path to violations_test.json
            max_tests: Optional limit on number of tests (for quick testing)
            
        Returns:
            Dict with all results and metrics
        """
        start_msg = "="*80 + "\n"
        start_msg += "STARTING END-TO-END SYSTEM TEST\n"
        start_msg += "="*80 + "\n"
        
        logger.info(start_msg)
        self.log_to_file(start_msg)
        
        # Load test cases
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        if max_tests:
            test_cases = test_cases[:max_tests]
            limit_msg = f"Running first {max_tests} tests only\n"
            logger.info(limit_msg)
            self.log_to_file(limit_msg)
        
        load_msg = f"Loaded {len(test_cases)} test cases\n"
        logger.info(load_msg)
        self.log_to_file(load_msg)
        
        # Run tests
        results = []
        exact_matches = 0
        retrieval_successes = 0
        total_time = 0
        
        for i, test_case in enumerate(test_cases):
            progress_msg = f"\n{'='*80}\nProgress: {i+1}/{len(test_cases)}\n{'='*80}"
            logger.info(progress_msg)
            self.log_to_file(progress_msg)
            
            result = self.test_single_question(test_case)
            results.append(result)
            
            if result['exact_match']:
                exact_matches += 1
            
            if result['retrieval_metrics']['expected_in_candidates']:
                retrieval_successes += 1
            
            total_time += result['response_time_seconds']
            
            # Log intermediate summary
            current_accuracy = (exact_matches / (i + 1) * 100) if i > 0 else (100 if exact_matches else 0)
            intermediate_msg = f"\nIntermediate Stats (after {i+1} tests):\n"
            intermediate_msg += f"  Accuracy: {current_accuracy:.2f}% ({exact_matches}/{i+1})\n"
            intermediate_msg += f"  Retrieval Success: {retrieval_successes}/{i+1}\n"
            self.log_to_file(intermediate_msg)
        
        # Calculate metrics
        total = len(results)
        exact_match_rate = (exact_matches / total * 100) if total > 0 else 0
        retrieval_success_rate = (retrieval_successes / total * 100) if total > 0 else 0
        avg_response_time = total_time / total if total > 0 else 0
        
        # Calculate MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for result in results:
            rank = result['retrieval_metrics'].get('expected_rank')
            if rank and rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
        
        # Calculate Recall@K
        recall_at_1 = sum(1 for r in results if r['retrieval_metrics'].get('expected_rank') == 1) / total * 100
        recall_at_3 = sum(1 for r in results if r['retrieval_metrics'].get('expected_rank') and r['retrieval_metrics']['expected_rank'] <= 3) / total * 100
        recall_at_5 = sum(1 for r in results if r['retrieval_metrics'].get('expected_rank') and r['retrieval_metrics']['expected_rank'] <= 5) / total * 100
        
        # Build summary
        summary = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': total,
                'test_file': test_file_path
            },
            'overall_metrics': {
                'exact_match_accuracy': round(exact_match_rate, 2),
                'exact_matches': exact_matches,
                'retrieval_success_rate': round(retrieval_success_rate, 2),
                'retrieval_successes': retrieval_successes,
                'avg_response_time_seconds': round(avg_response_time, 2),
                'total_time_seconds': round(total_time, 2)
            },
            'retrieval_metrics': {
                'mean_reciprocal_rank': round(mrr, 4),
                'recall_at_1': round(recall_at_1, 2),
                'recall_at_3': round(recall_at_3, 2),
                'recall_at_5': round(recall_at_5, 2)
            },
            'results': results
        }
        
        # Log final summary to file
        final_summary = "\n" + "="*80 + "\n"
        final_summary += "FINAL TEST SUMMARY\n"
        final_summary += "="*80 + "\n"
        final_summary += f"\nTest Run: {summary['test_info']['timestamp']}\n"
        final_summary += f"Total Tests: {summary['test_info']['total_tests']}\n"
        final_summary += f"\n--- OVERALL PERFORMANCE ---\n"
        final_summary += f"Exact Match Accuracy: {summary['overall_metrics']['exact_match_accuracy']}% "
        final_summary += f"({summary['overall_metrics']['exact_matches']}/{summary['test_info']['total_tests']})\n"
        final_summary += f"Retrieval Success Rate: {summary['overall_metrics']['retrieval_success_rate']}% "
        final_summary += f"({summary['overall_metrics']['retrieval_successes']}/{summary['test_info']['total_tests']})\n"
        final_summary += f"Average Response Time: {summary['overall_metrics']['avg_response_time_seconds']}s\n"
        final_summary += f"Total Test Time: {summary['overall_metrics']['total_time_seconds']}s\n"
        final_summary += f"\n--- RETRIEVAL METRICS ---\n"
        final_summary += f"Mean Reciprocal Rank (MRR): {summary['retrieval_metrics']['mean_reciprocal_rank']}\n"
        final_summary += f"Recall@1: {summary['retrieval_metrics']['recall_at_1']}%\n"
        final_summary += f"Recall@3: {summary['retrieval_metrics']['recall_at_3']}%\n"
        final_summary += f"Recall@5: {summary['retrieval_metrics']['recall_at_5']}%\n"
        final_summary += "="*80 + "\n"
        
        self.log_to_file(final_summary)
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        
        info = summary['test_info']
        metrics = summary['overall_metrics']
        retrieval = summary['retrieval_metrics']
        
        logger.info(f"\nTest Run: {info['timestamp']}")
        logger.info(f"Total Tests: {info['total_tests']}")
        
        logger.info("\n--- OVERALL PERFORMANCE ---")
        logger.info(f"Exact Match Accuracy: {metrics['exact_match_accuracy']}% ({metrics['exact_matches']}/{info['total_tests']})")
        logger.info(f"Retrieval Success Rate: {metrics['retrieval_success_rate']}% ({metrics['retrieval_successes']}/{info['total_tests']})")
        logger.info(f"Average Response Time: {metrics['avg_response_time_seconds']}s")
        logger.info(f"Total Test Time: {metrics['total_time_seconds']}s")
        
        logger.info("\n--- RETRIEVAL METRICS ---")
        logger.info(f"Mean Reciprocal Rank (MRR): {retrieval['mean_reciprocal_rank']}")
        logger.info(f"Recall@1: {retrieval['recall_at_1']}%")
        logger.info(f"Recall@3: {retrieval['recall_at_3']}%")
        logger.info(f"Recall@5: {retrieval['recall_at_5']}%")
        
        # Show failed cases
        failed_cases = [r for r in summary['results'] if not r['exact_match']]
        if failed_cases:
            logger.info("\n--- FAILED CASES ---")
            logger.info(f"Total Failed: {len(failed_cases)}")
            logger.info("-"*80)
            
            for result in failed_cases[:10]:  # Show first 10
                logger.info(f"\n✗ {result['test_id']}: {result['question']}")
                logger.info(f"  Expected: {result['expected_violation_id']}")
                logger.info(f"  Retrieved: {result['retrieved_violation_id']}")
                
                if result['retrieval_metrics']['expected_in_candidates']:
                    logger.info(f"  Note: Expected violation was at rank {result['retrieval_metrics']['expected_rank']}")
                else:
                    logger.info(f"  Note: Expected violation NOT in retrieval candidates")
                
                # Show what was retrieved instead
                top_candidates = result['retrieval_metrics'].get('all_candidates', [])[:5]
                if top_candidates:
                    logger.info(f"  Top candidate: {top_candidates[0]['violation_id']} (sim: {top_candidates[0]['similarity']})")
            
            if len(failed_cases) > 10:
                logger.info(f"\n... and {len(failed_cases) - 10} more failed cases")
        
        # Show successful cases summary
        success_cases = [r for r in summary['results'] if r['exact_match']]
        if success_cases:
            logger.info("\n--- SUCCESSFUL CASES ---")
            logger.info(f"Total Successful: {len(success_cases)}")
            
            # Show distribution of retrieval ranks for successful cases
            ranks = [r['retrieval_metrics'].get('expected_rank', 0) for r in success_cases if r['retrieval_metrics'].get('expected_rank')]
            if ranks:
                rank_1 = sum(1 for r in ranks if r == 1)
                rank_2_3 = sum(1 for r in ranks if 2 <= r <= 3)
                rank_4_5 = sum(1 for r in ranks if 4 <= r <= 5)
                rank_6_plus = sum(1 for r in ranks if r > 5)
                
                logger.info(f"  Rank 1: {rank_1} cases")
                logger.info(f"  Rank 2-3: {rank_2_3} cases")
                logger.info(f"  Rank 4-5: {rank_4_5} cases")
                logger.info(f"  Rank 6+: {rank_6_plus} cases")
        
        logger.info("\n" + "="*80)


def main():
    """Main function to run end-to-end tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='End-to-End System Testing')
    parser.add_argument(
        '--max-tests', 
        type=int, 
        default=None,
        help='Maximum number of tests to run (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/e2e_test_results.json',
        help='Output file path (default: data/e2e_test_results.json)'
    )
    parser.add_argument(
        '--log',
        type=str,
        default='data/e2e_test_log.txt',
        help='Log file path for real-time logging (default: data/e2e_test_log.txt)'
    )
    
    args = parser.parse_args()
    
    # Paths
    test_file = project_root / "data" / "violations_test.json"
    output_file = project_root / args.output
    log_file = project_root / args.log
    
    # Check if test file exists
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    # Initialize container
    logger.info("Initializing system components...")
    try:
        container = Container()
    except Exception as e:
        logger.error(f"Failed to initialize container: {e}")
        logger.error("Please check your .env configuration")
        return
    
    # Create tester with log file
    tester = SystemE2ETester(container, log_file=str(log_file))
    logger.info(f"Real-time logging to: {log_file}")
    
    # Run tests
    try:
        summary = tester.run_all_tests(str(test_file), max_tests=args.max_tests)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        return
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return
    
    # Print summary
    tester.print_summary(summary)
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Detailed log saved to: {log_file}")
    
    # Also save a CSV for easy analysis
    csv_file = output_file.with_suffix('.csv')
    try:
        import csv
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_id', 'question', 'expected_violation_id', 
                'retrieved_violation_id', 'exact_match',
                'expected_rank', 'expected_similarity',
                'response_time_seconds'
            ])
            writer.writeheader()
            for result in summary['results']:
                writer.writerow({
                    'test_id': result['test_id'],
                    'question': result['question'],
                    'expected_violation_id': result['expected_violation_id'],
                    'retrieved_violation_id': result['retrieved_violation_id'],
                    'exact_match': result['exact_match'],
                    'expected_rank': result['retrieval_metrics'].get('expected_rank', ''),
                    'expected_similarity': result['retrieval_metrics'].get('expected_similarity', ''),
                    'response_time_seconds': result['response_time_seconds']
                })
        logger.info(f"CSV results saved to: {csv_file}")
    except Exception as e:
        logger.warning(f"Could not save CSV: {e}")


if __name__ == "__main__":
    main()
