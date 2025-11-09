"""
Compare different conversion strategies and choose the best one
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def compare_strategies():
    """Compare all conversion strategies"""
    
    data_dir = Path('data/raw')
    strategies = ['strict', 'balanced', 'severity_aware', 'multi_label']
    
    results = []
    
    for strategy in strategies:
        file_path = data_dir / f'measuring_hate_speech_{strategy}.csv'
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        
        # Calculate class distribution
        class_dist = df['class'].value_counts(normalize=True).sort_index()
        
        # Calculate average confidence
        avg_conf = df['confidence'].mean() if 'confidence' in df.columns else 0
        
        results.append({
            'strategy': strategy,
            'total_samples': len(df),
            'hate_pct': class_dist.get(0, 0) * 100,
            'offensive_pct': class_dist.get(1, 0) * 100,
            'neither_pct': class_dist.get(2, 0) * 100,
            'avg_confidence': avg_conf
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("CONVERSION STRATEGY COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Class distribution
    ax = axes[0]
    x = range(len(results))
    width = 0.25
    
    ax.bar([i - width for i in x], comparison_df['hate_pct'], width, label='Hate', color='red', alpha=0.7)
    ax.bar(x, comparison_df['offensive_pct'], width, label='Offensive', color='orange', alpha=0.7)
    ax.bar([i + width for i in x], comparison_df['neither_pct'], width, label='Neither', color='green', alpha=0.7)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Percentage')
    ax.set_title('Class Distribution by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['strategy'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confidence
    ax = axes[1]
    ax.bar(comparison_df['strategy'], comparison_df['avg_confidence'], color='skyblue', alpha=0.7)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Average Confidence')
    ax.set_title('Average Confidence by Strategy')
    ax.set_xticklabels(comparison_df['strategy'], rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/conversion_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Comparison plot: results/conversion_strategy_comparison.png")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find most balanced
    comparison_df['balance_score'] = (
        abs(comparison_df['hate_pct'] - 15) +
        abs(comparison_df['offensive_pct'] - 55) +
        abs(comparison_df['neither_pct'] - 30)
    )
    
    best_balanced = comparison_df.loc[comparison_df['balance_score'].idxmin()]
    best_confidence = comparison_df.loc[comparison_df['avg_confidence'].idxmax()]
    
    print(f"\nMost Balanced Distribution: {best_balanced['strategy']}")
    print(f"  Hate: {best_balanced['hate_pct']:.1f}%")
    print(f"  Offensive: {best_balanced['offensive_pct']:.1f}%")
    print(f"  Neither: {best_balanced['neither_pct']:.1f}%")
    
    print(f"\nHighest Confidence: {best_confidence['strategy']}")
    print(f"  Average Confidence: {best_confidence['avg_confidence']:.3f}")
    
    print(f"\n[RECOMMENDED] Use '{best_balanced['strategy']}' for best balance")
    print(f"[RECOMMENDED] Use '{best_confidence['strategy']}' for highest confidence")

if __name__ == "__main__":
    compare_strategies()