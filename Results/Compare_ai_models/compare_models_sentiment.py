 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models_sentiment.py
---------------------------
Compare sentiment analysis results between different models (GPT-4o-mini vs GPT-5)
to evaluate consistency and differences in their annotations.
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Any, Tuple
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine

# Set up logging and style
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
plt.style.use('default')
sns.set_palette("husl")

def load_model_results(eval_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load sentiment results from different models."""
    models = {}
    
    for file_path in eval_dir.glob("*.json"):
        if "sentiment" in file_path.name:
            # Extract model name from filename
            if "4o_mini" in file_path.name:
                model_name = "GPT-4o-mini"
            elif "gpt5" in file_path.name:
                model_name = "GPT-5"
            else:
                model_name = file_path.stem.split('.')[-1]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                models[model_name] = data
                logging.info(f"Loaded {len(data)} messages from {model_name}")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")
    
    return models

def align_messages(models: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Align messages from different models by message_id."""
    # Create a combined dataframe
    aligned_data = []
    
    # Get all model names
    model_names = list(models.keys())
    if len(model_names) < 2:
        raise ValueError("Need at least 2 models to compare")
    
    # Use first model as reference
    reference_model = model_names[0]
    reference_data = {msg.get('message_id', idx): msg for idx, msg in enumerate(models[reference_model])}
    
    for msg_id, ref_msg in reference_data.items():
        row = {
            'message_id': msg_id,
            'body': ref_msg.get('body', ''),
            'timestamp': ref_msg.get('timestamp', ''),
        }
        
        # Add data from each model
        for model_name in model_names:
            model_data = models[model_name]
            # Find matching message by ID or index
            model_msg = None
            if isinstance(msg_id, str):
                model_msg = next((msg for msg in model_data if msg.get('message_id') == msg_id), None)
            else:
                if msg_id < len(model_data):
                    model_msg = model_data[msg_id]
            
            if model_msg:
                row[f'{model_name}_polarity'] = model_msg.get('polarity', 0)
                row[f'{model_name}_emotion_primary'] = model_msg.get('emotion_primary', 'neutral_info')
                row[f'{model_name}_emotion_summary'] = model_msg.get('emotion_summary', '')
                row[f'{model_name}_stress_score'] = model_msg.get('stress_score', 0)
                row[f'{model_name}_uncertainty_score'] = model_msg.get('uncertainty_score', 0)
                row[f'{model_name}_help_request'] = model_msg.get('help_request', False)
                row[f'{model_name}_helpfulness'] = model_msg.get('helpfulness', 0)
                row[f'{model_name}_gratitude'] = model_msg.get('gratitude', False)
                row[f'{model_name}_toxicity_score'] = model_msg.get('toxicity_score', 0)
                row[f'{model_name}_info_drop'] = model_msg.get('info_drop', False)
            else:
                # Fill with defaults if message not found
                for field in ['polarity', 'stress_score', 'uncertainty_score', 'helpfulness', 'toxicity_score']:
                    row[f'{model_name}_{field}'] = 0
                for field in ['emotion_primary', 'emotion_summary']:
                    row[f'{model_name}_{field}'] = 'neutral_info' if field == 'emotion_primary' else ''
                for field in ['help_request', 'gratitude', 'info_drop']:
                    row[f'{model_name}_{field}'] = False
        
        aligned_data.append(row)
    
    return pd.DataFrame(aligned_data)

def calculate_correlations(df: pd.DataFrame, model_names: List[str]) -> Dict[str, float]:
    """Calculate correlations between models for numeric fields."""
    correlations = {}
    
    numeric_fields = ['polarity', 'stress_score', 'uncertainty_score', 'helpfulness', 'toxicity_score']
    
    for field in numeric_fields:
        col1 = f'{model_names[0]}_{field}'
        col2 = f'{model_names[1]}_{field}'
        
        if col1 in df.columns and col2 in df.columns:
            # Remove any NaN values
            valid_data = df[[col1, col2]].dropna()
            if len(valid_data) > 1:
                pearson_corr, _ = pearsonr(valid_data[col1], valid_data[col2])
                spearman_corr, _ = spearmanr(valid_data[col1], valid_data[col2])
                correlations[field] = {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr,
                    'count': len(valid_data)
                }
    
    return correlations

def find_most_different_messages(df: pd.DataFrame, model_names: List[str], top_n: int = 10) -> List[Dict]:
    """Find messages with the largest differences in polarity between models."""
    pol_col1 = f'{model_names[0]}_polarity'
    pol_col2 = f'{model_names[1]}_polarity'
    
    df['polarity_diff'] = abs(df[pol_col1] - df[pol_col2])
    
    most_different = df.nlargest(top_n, 'polarity_diff')
    
    results = []
    for _, row in most_different.iterrows():
        results.append({
            'message_id': row['message_id'],
            'body': row['body'][:200] + "..." if len(row['body']) > 200 else row['body'],
            f'{model_names[0]}_polarity': row[pol_col1],
            f'{model_names[1]}_polarity': row[pol_col2],
            'difference': row['polarity_diff'],
            f'{model_names[0]}_emotion': row[f'{model_names[0]}_emotion_primary'],
            f'{model_names[1]}_emotion': row[f'{model_names[1]}_emotion_primary'],
        })
    
    return results

def calculate_basic_statistics(df: pd.DataFrame, model_names: List[str]) -> Dict[str, Dict]:
    """Calculate basic statistics for each model."""
    stats = {}
    
    numeric_fields = ['polarity', 'stress_score', 'uncertainty_score', 'helpfulness', 'toxicity_score']
    
    for model_name in model_names:
        model_stats = {}
        
        for field in numeric_fields:
            col_name = f'{model_name}_{field}'
            if col_name in df.columns:
                model_stats[field] = {
                    'mean': df[col_name].mean(),
                    'std': df[col_name].std(),
                    'min': df[col_name].min(),
                    'max': df[col_name].max(),
                    'median': df[col_name].median(),
                }
        
        # Emotion distribution
        emotion_col = f'{model_name}_emotion_primary'
        if emotion_col in df.columns:
            model_stats['emotion_distribution'] = df[emotion_col].value_counts().to_dict()
        
        # Boolean field counts
        bool_fields = ['help_request', 'gratitude', 'info_drop']
        for field in bool_fields:
            col_name = f'{model_name}_{field}'
            if col_name in df.columns:
                model_stats[f'{field}_count'] = df[col_name].sum()
                model_stats[f'{field}_percentage'] = (df[col_name].sum() / len(df)) * 100
        
        stats[model_name] = model_stats
    
    return stats

def create_visualizations(df: pd.DataFrame, model_names: List[str], output_dir: Path):
    """Create comprehensive visualizations comparing the models."""
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    # 1. Polarity comparison scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    # Scatter plot of polarities
    pol_col1 = f'{model_names[0]}_polarity'
    pol_col2 = f'{model_names[1]}_polarity'
    
    axes[0, 0].scatter(df[pol_col1], df[pol_col2], alpha=0.6, s=30)
    axes[0, 0].plot([-1, 1], [-1, 1], 'r--', label='Perfect Agreement')
    axes[0, 0].set_xlabel(f'{model_names[0]} Polarity')
    axes[0, 0].set_ylabel(f'{model_names[1]} Polarity')
    axes[0, 0].set_title('Polarity Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Polarity distribution comparison
    axes[0, 1].hist(df[pol_col1], bins=30, alpha=0.7, label=model_names[0], density=True)
    axes[0, 1].hist(df[pol_col2], bins=30, alpha=0.7, label=model_names[1], density=True)
    axes[0, 1].set_xlabel('Polarity')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Polarity Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Emotion distribution comparison by average polarity
    emotion_col1 = f'{model_names[0]}_emotion_primary'
    emotion_col2 = f'{model_names[1]}_emotion_primary'
    pol_col1 = f'{model_names[0]}_polarity'
    pol_col2 = f'{model_names[1]}_polarity'
    
    # Calculate average polarity by emotion for each model
    emotion_polarity1 = df.groupby(emotion_col1)[pol_col1].mean()
    emotion_polarity2 = df.groupby(emotion_col2)[pol_col2].mean()
    
    # Get top emotions by frequency for consistent ordering
    top_emotions = df[emotion_col1].value_counts().head(8).index
    
    # Filter to only include top emotions and ensure both models have data
    emotion_avg1 = emotion_polarity1.reindex(top_emotions).fillna(0)
    emotion_avg2 = emotion_polarity2.reindex(top_emotions).fillna(0)
    
    x = np.arange(len(emotion_avg1))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, emotion_avg1.values, width, label=model_names[0], alpha=0.8)
    axes[1, 0].bar(x + width/2, emotion_avg2.values, width, label=model_names[1], alpha=0.8)
    
    axes[1, 0].set_xlabel('Emotions')
    axes[1, 0].set_ylabel('Average Polarity')
    axes[1, 0].set_title('Average Polarity by Emotion')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(emotion_avg1.index, rotation=45, ha='right', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.3, top=0.92)
    
    # 4. Correlation heatmap of numeric fields
    numeric_fields = ['polarity', 'stress_score', 'uncertainty_score', 'helpfulness', 'toxicity_score']
    corr_data = []
    
    for field in numeric_fields:
        col1 = f'{model_names[0]}_{field}'
        col2 = f'{model_names[1]}_{field}'
        if col1 in df.columns and col2 in df.columns:
            valid_data = df[[col1, col2]].dropna()
            if len(valid_data) > 1:
                corr = pearsonr(valid_data[col1], valid_data[col2])[0]
                corr_data.append([field, corr])
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data, columns=['Field', 'Correlation'])
        bars = axes[1, 1].bar(corr_df['Field'], corr_df['Correlation'])
        axes[1, 1].set_xlabel('Sentiment Fields')
        axes[1, 1].set_ylabel('Pearson Correlation')
        axes[1, 1].set_title('Model Correlation by Field')
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color bars based on correlation strength
        for bar, corr in zip(bars, corr_df['Correlation']):
            if corr > 0.7:
                bar.set_color('green')
            elif corr > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    plt.tight_layout(pad=2.5)
    plt.savefig(output_dir / 'model_comparison_overview.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 5. Detailed numeric fields comparison
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Detailed Sentiment Fields Comparison', fontsize=13, fontweight='bold', y=0.98)
    
    numeric_fields = ['polarity', 'stress_score', 'uncertainty_score', 'helpfulness', 'toxicity_score']
    
    for i, field in enumerate(numeric_fields):
        row = i // 3
        col = i % 3
        
        col1 = f'{model_names[0]}_{field}'
        col2 = f'{model_names[1]}_{field}'
        
        if col1 in df.columns and col2 in df.columns:
            axes[row, col].scatter(df[col1], df[col2], alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(df[col1].min(), df[col2].min())
            max_val = max(df[col1].max(), df[col2].max())
            axes[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            
            axes[row, col].set_xlabel(f'{model_names[0]} {field}')
            axes[row, col].set_ylabel(f'{model_names[1]} {field}')
            axes[row, col].set_title(f'{field.replace("_", " ").title()}')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add correlation text
            valid_data = df[[col1, col2]].dropna()
            if len(valid_data) > 1:
                corr = pearsonr(valid_data[col1], valid_data[col2])[0]
                axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', 
                                  transform=axes[row, col].transAxes, 
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove the empty subplot
    if len(numeric_fields) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    plt.savefig(output_dir / 'detailed_fields_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def create_summary_report(stats: Dict, correlations: Dict, most_different: List[Dict], 
                         model_names: List[str]) -> str:
    """Create a comprehensive summary report."""
    report = [f"# Model Comparison Report: {model_names[0]} vs {model_names[1]}\n"]
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    # Overall correlations
    avg_correlation = np.mean([corr['pearson'] for corr in correlations.values() if not np.isnan(corr['pearson'])])
    report.append(f"**Average Correlation:** {avg_correlation:.3f}")
    
    if avg_correlation > 0.7:
        agreement_level = "High"
    elif avg_correlation > 0.4:
        agreement_level = "Moderate"
    else:
        agreement_level = "Low"
    
    report.append(f"**Agreement Level:** {agreement_level}\n")
    
    # Detailed Statistics Comparison
    report.append("## Detailed Statistics Comparison\n")
    
    report.append("### Polarity Analysis")
    report.append("| Model | Mean | Std | Min | Max | Median |")
    report.append("|-------|------|-----|-----|-----|--------|")
    
    for model_name in model_names:
        pol_stats = stats[model_name]['polarity']
        report.append(f"| {model_name} | {pol_stats['mean']:.3f} | {pol_stats['std']:.3f} | "
                     f"{pol_stats['min']:.3f} | {pol_stats['max']:.3f} | {pol_stats['median']:.3f} |")
    
    # Correlation Analysis
    report.append("\n### Field Correlations")
    report.append("| Field | Pearson | Spearman | Interpretation |")
    report.append("|-------|---------|----------|---------------|")
    
    for field, corr_data in correlations.items():
        pearson = corr_data['pearson']
        spearman = corr_data['spearman']
        
        if pearson > 0.7:
            interpretation = "Strong Agreement"
        elif pearson > 0.4:
            interpretation = "Moderate Agreement"
        elif pearson > 0.1:
            interpretation = "Weak Agreement"
        else:
            interpretation = "No Agreement"
        
        report.append(f"| {field.replace('_', ' ').title()} | {pearson:.3f} | {spearman:.3f} | {interpretation} |")
    
    # Most Different Messages
    report.append(f"\n### Top 5 Most Different Messages")
    report.append("| Message Preview | " + " | ".join([f"{model} Polarity" for model in model_names]) + " | Difference |")
    report.append("|" + "|".join(["-" * 20 for _ in range(len(model_names) + 2)]) + "|")
    
    for msg in most_different[:5]:
        preview = msg['body'].replace('\n', ' ').replace('|', '\\|')[:50] + "..."
        polarities = " | ".join([f"{msg[f'{model}_polarity']:.3f}" for model in model_names])
        report.append(f"| {preview} | {polarities} | {msg['difference']:.3f} |")
    
    # Key Insights
    report.append("\n## Key Insights\n")
    
    # Polarity differences
    pol_mean_diff = abs(stats[model_names[0]]['polarity']['mean'] - stats[model_names[1]]['polarity']['mean'])
    report.append(f"1. **Polarity Mean Difference:** {pol_mean_diff:.3f}")
    
    if pol_mean_diff < 0.1:
        report.append("   - Models show very similar average sentiment")
    elif pol_mean_diff < 0.3:
        report.append("   - Models show moderate difference in average sentiment")
    else:
        report.append("   - Models show significant difference in average sentiment")
    
    # Strongest and weakest correlations
    best_corr_field = max(correlations.items(), key=lambda x: x[1]['pearson'])[0]
    worst_corr_field = min(correlations.items(), key=lambda x: x[1]['pearson'])[0]
    
    report.append(f"\n2. **Strongest Agreement:** {best_corr_field.replace('_', ' ').title()} "
                 f"(r = {correlations[best_corr_field]['pearson']:.3f})")
    report.append(f"3. **Weakest Agreement:** {worst_corr_field.replace('_', ' ').title()} "
                 f"(r = {correlations[worst_corr_field]['pearson']:.3f})")
    
    # Range analysis
    pol_range_diff = abs((stats[model_names[0]]['polarity']['max'] - stats[model_names[0]]['polarity']['min']) - 
                        (stats[model_names[1]]['polarity']['max'] - stats[model_names[1]]['polarity']['min']))
    
    report.append(f"\n4. **Polarity Range Difference:** {pol_range_diff:.3f}")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Compare sentiment analysis results between models')
    parser.add_argument('--eval-dir', default='./Evaluation', help='Directory containing evaluation files')
    parser.add_argument('--output', default='model_comparison', help='Output file prefix')
    parser.add_argument('--top-different', type=int, default=10, help='Number of most different messages to show')
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    output_path = Path(args.output)
    
    if not eval_dir.exists():
        logging.error(f"Evaluation directory does not exist: {eval_dir}")
        return
    
    # Load model results
    logging.info(f"Loading model results from {eval_dir}")
    models = load_model_results(eval_dir)
    
    if len(models) < 2:
        logging.error("Need at least 2 models to compare")
        return
    
    model_names = list(models.keys())
    logging.info(f"Comparing models: {model_names}")
    
    # Align messages
    logging.info("Aligning messages between models...")
    df = align_messages(models)
    logging.info(f"Aligned {len(df)} messages")
    
    # Calculate statistics
    logging.info("Calculating statistics...")
    stats = calculate_basic_statistics(df, model_names)
    correlations = calculate_correlations(df, model_names)
    most_different = find_most_different_messages(df, model_names, args.top_different)
    
    # Create visualizations
    logging.info("Creating visualizations...")
    create_visualizations(df, model_names, Path('./plots'))
    
    # Generate report
    logging.info("Generating report...")
    report = create_summary_report(stats, correlations, most_different, model_names)
    
    # Save results
    results = {
        'models_compared': model_names,
        'total_messages': len(df),
        'statistics': stats,
        'correlations': correlations,
        'most_different_messages': most_different[:args.top_different],
        'summary_report': report
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    # Save JSON
    with open(f'{output_path}.json', 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    
    # Save report
    with open(f'{output_path}_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save CSV for further analysis
    df.to_csv(f'{output_path}_aligned_data.csv', index=False, encoding='utf-8')
    
    logging.info(f"Results saved to {output_path}.json, {output_path}_report.md, and {output_path}_aligned_data.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"Models: {' vs '.join(model_names)}")
    print(f"Messages analyzed: {len(df)}")
    print(f"Average correlation: {np.mean([corr['pearson'] for corr in correlations.values() if not np.isnan(corr['pearson'])]):.3f}")
    
    print(f"\nPolarity Statistics:")
    for model_name in model_names:
        pol_stats = stats[model_name]['polarity']
        print(f"  {model_name}: mean={pol_stats['mean']:.3f}, std={pol_stats['std']:.3f}")
    
    print(f"\nField Correlations:")
    for field, corr_data in correlations.items():
        print(f"  {field}: r={corr_data['pearson']:.3f}")
    
    print(f"\nTop 3 Most Different Messages:")
    for i, msg in enumerate(most_different[:3], 1):
        print(f"  {i}. Difference: {msg['difference']:.3f}")
        print(f"     Preview: {msg['body'][:100]}...")
        for model in model_names:
            print(f"     {model}: {msg[f'{model}_polarity']:.3f} ({msg[f'{model}_emotion']})")
        print()

if __name__ == "__main__":
    main()