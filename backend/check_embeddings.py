#!/usr/bin/env python3
"""
Embedding Health Checker

A utility script to check embedding diversity and detect model collapse.
This script can be run on any training artifacts directory to assess model health.

Usage:
    python backend/check_embeddings.py [artifacts_path]
    python backend/check_embeddings.py artifacts/two_tower_run_20250619_174945
"""

import torch
import numpy as np
import pickle
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import get_best_device


class EmbeddingHealthChecker:
    """Comprehensive embedding health analysis."""
    
    def __init__(self, artifacts_path: str):
        self.artifacts_path = artifacts_path
        self.device = get_best_device()
        self.doc_embeddings = None
        self.query_embeddings = None
        self.doc_to_idx = None
        self.query_to_idx = None
        
    def load_embeddings(self):
        """Load embeddings and mappings from artifacts directory."""
        print(f"📂 Loading embeddings from: {self.artifacts_path}")
        
        try:
            # Load embeddings
            doc_path = f"{self.artifacts_path}/document_embeddings.npy"
            query_path = f"{self.artifacts_path}/query_embeddings.npy"
            
            self.doc_embeddings = torch.from_numpy(np.load(doc_path)).to(self.device)
            self.query_embeddings = torch.from_numpy(np.load(query_path)).to(self.device)
            
            # Load mappings
            with open(f"{self.artifacts_path}/doc_to_idx.pkl", 'rb') as f:
                self.doc_to_idx = pickle.load(f)
            with open(f"{self.artifacts_path}/query_to_idx.pkl", 'rb') as f:
                self.query_to_idx = pickle.load(f)
                
            print(f"  ✅ Loaded {len(self.doc_embeddings):,} document embeddings")
            print(f"  ✅ Loaded {len(self.query_embeddings):,} query embeddings")
            return True
            
        except FileNotFoundError as e:
            print(f"  ❌ Error loading embeddings: {e}")
            return False
    
    def basic_stats(self):
        """Print basic embedding statistics."""
        print(f"\n📊 BASIC STATISTICS")
        print("─" * 50)
        
        print(f"  Document embeddings shape: {self.doc_embeddings.shape}")
        print(f"  Query embeddings shape: {self.query_embeddings.shape}")
        print(f"  Embedding dimension: {self.doc_embeddings.shape[1]}")
        print(f"  Data type: {self.doc_embeddings.dtype}")
        
        # Check norms (should be 1.0 if normalized)
        doc_norms = torch.norm(self.doc_embeddings, dim=1)
        query_norms = torch.norm(self.query_embeddings, dim=1)
        
        print(f"\n📏 EMBEDDING NORMS:")
        print(f"  Doc norms  - min: {doc_norms.min():.6f}, max: {doc_norms.max():.6f}, mean: {doc_norms.mean():.6f}")
        print(f"  Query norms - min: {query_norms.min():.6f}, max: {query_norms.max():.6f}, mean: {query_norms.mean():.6f}")
        
        normalized = (doc_norms.min() > 0.99 and doc_norms.max() < 1.01 and 
                     query_norms.min() > 0.99 and query_norms.max() < 1.01)
        print(f"  Normalized: {'✅ Yes' if normalized else '❌ No'}")
    
    def diversity_analysis(self, num_samples=20):
        """Analyze embedding diversity by computing pairwise similarities."""
        print(f"\n🔍 DIVERSITY ANALYSIS")
        print("─" * 50)
        
        # Document diversity
        if len(self.doc_embeddings) > 1:
            n_samples = min(num_samples, len(self.doc_embeddings))
            random_indices = torch.randperm(len(self.doc_embeddings))[:n_samples]
            
            doc_similarities = []
            for i in range(len(random_indices)):
                for j in range(i+1, len(random_indices)):
                    sim = torch.nn.functional.cosine_similarity(
                        self.doc_embeddings[random_indices[i]].unsqueeze(0), 
                        self.doc_embeddings[random_indices[j]].unsqueeze(0)
                    )
                    doc_similarities.append(sim.item())
            
            doc_sim_mean = np.mean(doc_similarities)
            doc_sim_std = np.std(doc_similarities)
            doc_sim_min = np.min(doc_similarities)
            doc_sim_max = np.max(doc_similarities)
            
            print(f"  Document similarity ({n_samples} random pairs):")
            print(f"    Mean: {doc_sim_mean:.6f}")
            print(f"    Std:  {doc_sim_std:.6f}")
            print(f"    Min:  {doc_sim_min:.6f}")
            print(f"    Max:  {doc_sim_max:.6f}")
            
            # Classify diversity level
            if doc_sim_mean > 0.995:
                print(f"    Status: 🚨 SEVERE COLLAPSE (mean > 0.995)")
            elif doc_sim_mean > 0.99:
                print(f"    Status: ⚠️  HIGH COLLAPSE (mean > 0.99)")
            elif doc_sim_mean > 0.95:
                print(f"    Status: ⚠️  MODERATE COLLAPSE (mean > 0.95)")
            elif doc_sim_mean > 0.8:
                print(f"    Status: 🟡 LOW DIVERSITY (mean > 0.8)")
            else:
                print(f"    Status: ✅ HEALTHY DIVERSITY")
        
        # Query diversity  
        if len(self.query_embeddings) > 1:
            n_samples = min(num_samples, len(self.query_embeddings))
            random_indices = torch.randperm(len(self.query_embeddings))[:n_samples]
            
            query_similarities = []
            for i in range(len(random_indices)):
                for j in range(i+1, len(random_indices)):
                    sim = torch.nn.functional.cosine_similarity(
                        self.query_embeddings[random_indices[i]].unsqueeze(0), 
                        self.query_embeddings[random_indices[j]].unsqueeze(0)
                    )
                    query_similarities.append(sim.item())
            
            query_sim_mean = np.mean(query_similarities)
            query_sim_std = np.std(query_similarities)
            query_sim_min = np.min(query_similarities)
            query_sim_max = np.max(query_similarities)
            
            print(f"\n  Query similarity ({n_samples} random pairs):")
            print(f"    Mean: {query_sim_mean:.6f}")
            print(f"    Std:  {query_sim_std:.6f}")
            print(f"    Min:  {query_sim_min:.6f}")
            print(f"    Max:  {query_sim_max:.6f}")
            
            # Classify diversity level
            if query_sim_mean > 0.995:
                print(f"    Status: 🚨 SEVERE COLLAPSE (mean > 0.995)")
            elif query_sim_mean > 0.99:
                print(f"    Status: ⚠️  HIGH COLLAPSE (mean > 0.99)")
            elif query_sim_mean > 0.95:
                print(f"    Status: ⚠️  MODERATE COLLAPSE (mean > 0.95)")
            elif query_sim_mean > 0.8:
                print(f"    Status: 🟡 LOW DIVERSITY (mean > 0.8)")
            else:
                print(f"    Status: ✅ HEALTHY DIVERSITY")
                
            return doc_sim_mean, query_sim_mean
        
        return None, None
    
    def variance_analysis(self):
        """Analyze embedding variance across dimensions."""
        print(f"\n📈 VARIANCE ANALYSIS")
        print("─" * 50)
        
        # Per-dimension variance
        doc_var_per_dim = torch.var(self.doc_embeddings, dim=0)
        query_var_per_dim = torch.var(self.query_embeddings, dim=0)
        
        # Overall statistics
        doc_total_var = torch.var(self.doc_embeddings)
        query_total_var = torch.var(self.query_embeddings)
        
        doc_mean_var = doc_var_per_dim.mean()
        query_mean_var = query_var_per_dim.mean()
        
        print(f"  Document embeddings:")
        print(f"    Total variance: {doc_total_var:.8f}")
        print(f"    Mean per-dim variance: {doc_mean_var:.8f}")
        print(f"    Min per-dim variance: {doc_var_per_dim.min():.8f}")
        print(f"    Max per-dim variance: {doc_var_per_dim.max():.8f}")
        
        print(f"\n  Query embeddings:")
        print(f"    Total variance: {query_total_var:.8f}")
        print(f"    Mean per-dim variance: {query_mean_var:.8f}")
        print(f"    Min per-dim variance: {query_var_per_dim.min():.8f}")
        print(f"    Max per-dim variance: {query_var_per_dim.max():.8f}")
        
        # Health check
        if doc_total_var < 1e-6:
            print(f"    Doc Status: 🚨 ZERO VARIANCE - COMPLETE COLLAPSE")
        elif doc_total_var < 1e-4:
            print(f"    Doc Status: ⚠️  VERY LOW VARIANCE")
        elif doc_total_var < 1e-2:
            print(f"    Doc Status: 🟡 LOW VARIANCE")
        else:
            print(f"    Doc Status: ✅ HEALTHY VARIANCE")
            
        if query_total_var < 1e-6:
            print(f"    Query Status: 🚨 ZERO VARIANCE - COMPLETE COLLAPSE")
        elif query_total_var < 1e-4:
            print(f"    Query Status: ⚠️  VERY LOW VARIANCE")
        elif query_total_var < 1e-2:
            print(f"    Query Status: 🟡 LOW VARIANCE")
        else:
            print(f"    Query Status: ✅ HEALTHY VARIANCE")
        
        return doc_total_var, query_total_var
    
    def cross_modal_analysis(self, num_samples=10):
        """Analyze query-document similarities."""
        print(f"\n🔄 CROSS-MODAL ANALYSIS")
        print("─" * 50)
        
        n_queries = min(num_samples, len(self.query_embeddings))
        n_docs = min(num_samples, len(self.doc_embeddings))
        
        query_indices = torch.randperm(len(self.query_embeddings))[:n_queries]
        doc_indices = torch.randperm(len(self.doc_embeddings))[:n_docs]
        
        cross_similarities = []
        for q_idx in query_indices:
            for d_idx in doc_indices:
                sim = torch.nn.functional.cosine_similarity(
                    self.query_embeddings[q_idx].unsqueeze(0),
                    self.doc_embeddings[d_idx].unsqueeze(0)
                )
                cross_similarities.append(sim.item())
        
        cross_sim_mean = np.mean(cross_similarities)
        cross_sim_std = np.std(cross_similarities)
        cross_sim_min = np.min(cross_similarities)
        cross_sim_max = np.max(cross_similarities)
        
        print(f"  Query-Document similarity ({n_queries}x{n_docs} pairs):")
        print(f"    Mean: {cross_sim_mean:.6f}")
        print(f"    Std:  {cross_sim_std:.6f}")
        print(f"    Min:  {cross_sim_min:.6f}")
        print(f"    Max:  {cross_sim_max:.6f}")
        
        # Ideally, cross-modal similarities should have good spread
        if cross_sim_std < 0.01:
            print(f"    Status: ⚠️  LOW CROSS-MODAL DIVERSITY (std < 0.01)")
        else:
            print(f"    Status: ✅ GOOD CROSS-MODAL DIVERSITY")
            
        return cross_sim_mean, cross_sim_std
    
    def overall_health_assessment(self, doc_sim_mean, query_sim_mean, doc_var, query_var):
        """Provide overall health assessment."""
        print(f"\n🏥 OVERALL HEALTH ASSESSMENT")
        print("=" * 50)
        
        issues = []
        
        # Check for model collapse
        if doc_sim_mean and doc_sim_mean > 0.99:
            issues.append(f"Document embeddings too similar (mean: {doc_sim_mean:.4f})")
        if query_sim_mean and query_sim_mean > 0.99:
            issues.append(f"Query embeddings too similar (mean: {query_sim_mean:.4f})")
            
        # Check for low variance
        if doc_var < 1e-4:
            issues.append(f"Document embeddings have very low variance ({doc_var:.8f})")
        if query_var < 1e-4:
            issues.append(f"Query embeddings have very low variance ({query_var:.8f})")
        
        # Overall assessment
        if not issues:
            print("  🎉 EMBEDDINGS LOOK HEALTHY!")
            print("     • Good diversity in both query and document embeddings")
            print("     • Sufficient variance across dimensions")
            print("     • Model should perform well for retrieval tasks")
            return True
        else:
            print("  🚨 ISSUES DETECTED:")
            for issue in issues:
                print(f"     • {issue}")
            
            print(f"\n  💡 RECOMMENDATIONS:")
            print(f"     • Retrain with lower learning rate")
            print(f"     • Add regularization (dropout)")
            print(f"     • Use smaller batch sizes")
            print(f"     • Check training data diversity")
            print(f"     • Monitor training loss curves")
            return False
    
    def run_full_analysis(self):
        """Run complete embedding health analysis."""
        print("🔍 EMBEDDING HEALTH CHECK")
        print("=" * 80)
        
        if not self.load_embeddings():
            return False
        
        self.basic_stats()
        doc_sim_mean, query_sim_mean = self.diversity_analysis()
        doc_var, query_var = self.variance_analysis()
        self.cross_modal_analysis()
        
        is_healthy = self.overall_health_assessment(doc_sim_mean, query_sim_mean, doc_var, query_var)
        
        print("\n" + "=" * 80)
        return is_healthy


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Check embedding health and detect model collapse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backend/check_embeddings.py
  python backend/check_embeddings.py artifacts/two_tower_run_20250619_163538
  python backend/check_embeddings.py artifacts/latest_run
        """
    )
    
    parser.add_argument(
        'artifacts_path', 
        nargs='?', 
        default='artifacts/two_tower_run_20250619_163538',
        help='Path to artifacts directory (default: artifacts/two_tower_run_20250619_163538)'
    )
    
    args = parser.parse_args()
    
    # Check if path exists
    if not Path(args.artifacts_path).exists():
        print(f"❌ Error: Path '{args.artifacts_path}' does not exist!")
        print(f"💡 Make sure you've run training and generated embeddings.")
        return 1
    
    # Run analysis
    checker = EmbeddingHealthChecker(args.artifacts_path)
    is_healthy = checker.run_full_analysis()
    
    return 0 if is_healthy else 1


if __name__ == "__main__":
    exit(main()) 