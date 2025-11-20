#!/usr/bin/env python3
"""
Generate result figures from neuropsych VLM benchmark CSV files.
This aims to replicate the analysis done in the original paper (Supplementary Figure S2 and S3).

This script processes multiple CSV files containing VLM benchmark results and generates
comparative plots including:
- Performance relative to human norms by stage
- Performance relative to human norms by subtask grouping
- Statistical comparisons using permutation tests

Note:
    This automated plotting script has been validated using the original paper's data for supplementary Figure S2 and S3.
    csv files used can be found in https://osf.io/ysxvg/files/ck5sr.

To instantly run the script, replace the csv files in the RESULT_CSVS list with your own csv files at the bottom of the script.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from typing import List, Dict


# Set random seeds for reproducibility
np.random.seed(42)

# Configuration
STAGES = ['low', 'mid', 'high']
SUBTASK_GROUPING = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SUBTASK_GROUPING_LABELS = [
    'Simple Element Judgements',
    'Figure Judgements',
    'Occlusion and Overlap',
    'Property Biases',
    'Robustness Across Different Image cues',
    'Robustness Across Different Configurations',
    'Recognition in Visually Straightforward Circumstances',
    'View Invariance',
    'Semantic Association and Categorisation'
]

# Color palettes
try:
    from figures_utils.plots_colouring import three_colors, nine_colors
except ImportError:
    print("Warning: plots_colouring.py not found. Using default color palettes.")
    three_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    nine_colors = sns.color_palette("tab10", 9).as_hex()


def extract_model_name(filepath: str) -> str:
    """Extract model name from filepath."""
    filename = Path(filepath).stem
    # Remove 'results_' prefix if present
    if filename.startswith('results_'):
        filename = filename[8:]
    return filename


def load_baseline_data(baseline_path: str) -> pd.DataFrame:
    """Load and prepare baseline data with metadata and old model scores."""
    data = pd.read_csv(baseline_path)
    
    # Select and rename columns
    columns_to_keep = [
        "task", "display_name", "subtask_grouping", "stage",
        "GPT_performance_score", "Claude_performance_score", "Gemini_performance_score",
        "full_score", "normative_score", "normative_SD"
    ]
    
    # This is the data from the original paper (for the 31 open-source subset)
    data = data[columns_to_keep]
    data = data.rename(columns={
        "GPT_performance_score": "old_GPT_score",
        "Claude_performance_score": "old_Claude_score",
        "Gemini_performance_score": "old_Gemini_score"
    })
    
    return data


def load_result_data(result_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Load result CSV files and return dict of DataFrames."""
    result_data = {}
    
    for path in result_paths:
        model_name = extract_model_name(path)
        df = pd.read_csv(path)
        
        if 'task' in df.columns and 'raw_score' in df.columns:
            result_data[model_name] = df[['task', 'raw_score']].rename(
                columns={'raw_score': f'new_{model_name}_score'}
            )
        else:
            print(f"Warning: {path} does not contain required columns 'task' and 'raw_score'")
    
    return result_data


def merge_data(baseline: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge baseline and result data."""
    data = baseline.copy()
    
    # Merge each result dataset
    for model_name, result_df in results.items():
        data = pd.merge(data, result_df, on='task', how='left')
    
    return data


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data by setting categorical columns and computing percentages."""
    # Set categorical columns
    data['stage'] = pd.Categorical(data['stage'], categories=STAGES, ordered=True)
    data['subtask_grouping'] = pd.Categorical(
        data['subtask_grouping'],
        categories=SUBTASK_GROUPING,
        ordered=True
    )
    data['subtask_grouping'] = data['subtask_grouping'].cat.rename_categories(SUBTASK_GROUPING_LABELS)
    
    # Sort data
    data = data.sort_values(['stage', 'subtask_grouping'])
    
    # Convert score columns to numeric
    score_columns = [col for col in data.columns if '_score' in col or 'normative' in col]
    data[score_columns] = data[score_columns].apply(pd.to_numeric, errors='coerce')
    
    # Calculate percentages
    for col in data.columns:
        if '_score' in col and col != 'full_score':
            percent_col = col.replace('_score', '_score_percent')
            data[percent_col] = (data[col] / data['full_score']) * 100
    
    data['normative_percent'] = (data['normative_score'] / data['full_score']) * 100
    data['normative_sd_percent'] = (data['normative_SD'] / data['full_score']) * 100
    
    # Calculate relative to human performance
    for col in data.columns:
        if '_score_percent' in col and 'normative' not in col:
            human_col = col.replace('_score_percent', '_human')
            data[human_col] = (data[col] - data['normative_percent']) / data['normative_percent']
    
    # Calculate significance flags
    for col in data.columns:
        if '_score' in col and 'percent' not in col and 'full' not in col and 'normative' not in col:
            prefix = col.replace('_score', '')
            data[f'{prefix}_sig_below'] = (
                (data[col] - data['normative_score']) / data['normative_SD']
            ) <= -2
            data[f'{prefix}_sig_above'] = (
                (data[col] - data['normative_score']) / data['normative_SD']
            ) >= 2
    
    return data




def permutation_test(data: pd.DataFrame, model_columns: List[str], n_permutations: int = 10000) -> Dict:
    """Perform permutation test comparing stages."""
    # Calculate average across models
    data['VLMs_relative_to_human'] = data[model_columns].mean(axis=1)
    
    # Extract data by stage
    low_data = data.loc[data['stage'] == 'low', 'VLMs_relative_to_human']
    mid_data = data.loc[data['stage'] == 'mid', 'VLMs_relative_to_human']
    high_data = data.loc[data['stage'] == 'high', 'VLMs_relative_to_human']
    
    # Calculate observed mean differences
    mean_diff_low_mid = low_data.mean() - mid_data.mean()
    mean_diff_low_high = low_data.mean() - high_data.mean()
    mean_diff_mid_high = mid_data.mean() - high_data.mean()
    
    print(f'\nMean Difference Low vs Mid: {mean_diff_low_mid:.2f}')
    print(f'Mean Difference Low vs High: {mean_diff_low_high:.2f}')
    print(f'Mean Difference Mid vs High: {mean_diff_mid_high:.2f}')
    
    # Permutation test
    permuted_mean_diff_low_mid = []
    permuted_mean_diff_low_high = []
    permuted_mean_diff_mid_high = []
    
    for _ in range(n_permutations):
        all_data = pd.concat([low_data, mid_data, high_data])
        all_data_shuffled = all_data.sample(frac=1).reset_index(drop=True)
        low_shuffled = all_data_shuffled[:len(low_data)]
        mid_shuffled = all_data_shuffled[len(low_data):len(low_data)+len(mid_data)]
        high_shuffled = all_data_shuffled[len(low_data)+len(mid_data):]
        
        permuted_mean_diff_low_mid.append(low_shuffled.mean() - mid_shuffled.mean())
        permuted_mean_diff_low_high.append(low_shuffled.mean() - high_shuffled.mean())
        permuted_mean_diff_mid_high.append(mid_shuffled.mean() - high_shuffled.mean())
    
    # Calculate z-scores and p-values
    z_score_low_mid = (mean_diff_low_mid - np.mean(permuted_mean_diff_low_mid)) / np.std(permuted_mean_diff_low_mid)
    z_score_low_high = (mean_diff_low_high - np.mean(permuted_mean_diff_low_high)) / np.std(permuted_mean_diff_low_high)
    z_score_mid_high = (mean_diff_mid_high - np.mean(permuted_mean_diff_mid_high)) / np.std(permuted_mean_diff_mid_high)
    
    pvalue_low_mid = scipy.stats.norm.sf(abs(z_score_low_mid)) * 2
    pvalue_low_high = scipy.stats.norm.sf(abs(z_score_low_high)) * 2
    pvalue_mid_high = scipy.stats.norm.sf(abs(z_score_mid_high)) * 2
    
    # Bonferroni correction
    alpha = 0.05
    n_comparisons = 3
    
    print(f'\nLow vs Mid - Significant after Bonferroni correction? {pvalue_low_mid < alpha/n_comparisons} (p={pvalue_low_mid:.4f})')
    print(f'Low vs High - Significant after Bonferroni correction? {pvalue_low_high < alpha/n_comparisons} (p={pvalue_low_high:.4f})')
    print(f'Mid vs High - Significant after Bonferroni correction? {pvalue_mid_high < alpha/n_comparisons} (p={pvalue_mid_high:.4f})')
    
    return {
        'mean_diffs': (mean_diff_low_mid, mean_diff_low_high, mean_diff_mid_high),
        'p_values': (pvalue_low_mid, pvalue_low_high, pvalue_mid_high)
    }


def plot_stage_comparison(data: pd.DataFrame, model_columns: List[str], output_path: str, perm_test_results: Dict = None):
    """Generate boxplot comparing VLM performance across stages with significance annotations."""
    data['VLMs_relative_to_human'] = data[model_columns].mean(axis=1)
    
    model_palette = sns.color_palette(three_colors)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.boxplot(
        x='stage',
        y='VLMs_relative_to_human',
        data=data,
        hue='stage',
        palette=model_palette,
        width=0.6,
        ax=ax,
        legend=False
    )
    sns.despine()
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Low', 'Mid', 'High'])
    ax.set_xlabel('Stage')
    ax.set_ylabel('Averaged VLMs Performance Relative to Human')
    ax.set_ylim(-1, 0.5)
    
    # Add significance annotations if permutation test results are provided
    if perm_test_results:
        p_values = perm_test_results['p_values']
        n_comparisons = 3
        
        # Get y-axis limits for positioning
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        # Annotation positions
        h = y_max - 0.05 * y_range  # Height for lines
        
        # Helper function to get significance marker (Bonferroni-corrected p-values)
        def get_sig_marker(p_val, n_comp):
            if p_val < 0.001 / n_comp:
                return '***'
            elif p_val < 0.01 / n_comp:
                return '**'
            elif p_val < 0.05 / n_comp:
                return '*'
            return None
        
        # Low vs High
        sig_marker = get_sig_marker(p_values[1], n_comparisons)
        if sig_marker:
            ax.plot([0, 0, 2, 2], [h, h + 0.02, h + 0.02, h], lw=1.5, c='black')
            ax.text(1, h + 0.03, sig_marker, ha='center', va='bottom', color='black', fontsize=14)
        
        # Mid vs High
        sig_marker = get_sig_marker(p_values[2], n_comparisons)
        if sig_marker:
            ax.plot([1, 1, 2, 2], [h - 0.08, h - 0.06, h - 0.06, h - 0.08], lw=1.5, c='black')
            ax.text(1.5, h - 0.05, sig_marker, ha='center', va='bottom', color='black', fontsize=14)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stage comparison plot to {output_path}")


def adjust_brightness(color, factor):
    """Adjust brightness of a color."""
    return tuple(min(1, c * factor) for c in color)


def plot_subtask_comparison(data: pd.DataFrame, model_columns: List[str], output_path: str):
    """Generate horizontal bar plot comparing VLM performance by subtask."""
    fig, ax = plt.subplots(figsize=(5, 15))
    
    subtask_palette = sns.color_palette(nine_colors)
    color_dict = dict(zip(SUBTASK_GROUPING_LABELS, subtask_palette))
    
    # Generate brightness factors dynamically based on number of models
    num_models = len(model_columns)
    brightness_factors = [1.0 - (i * 0.3) for i in range(num_models)]
    bar_width = 0.3
    y_tick_positions = []
    y_tick_labels = []
    bar_count = 0
    
    for subtask, group in data.groupby('subtask_grouping', observed=True):
        model_colors = {
            model: adjust_brightness(color_dict[subtask], factor)
            for model, factor in zip(model_columns, brightness_factors)
        }
        
        y_positions = np.arange(bar_count, bar_count + len(group))
        bar_count += len(group)
        
        y_tick_positions.extend(y_positions + bar_width)
        y_tick_labels.extend(group['display_name'])
        
        for i, model in enumerate(model_columns):
            bars = ax.barh(
                y_positions + i * bar_width,
                group[model],
                bar_width,
                label=f"{subtask} - {model.split('_')[0]}",
                color=model_colors[model],
            )
            
            # Add significance markers
            for bar_idx, bar in enumerate(bars):
                row = group.iloc[bar_idx]
                prefix = '_'.join(model.split('_')[:2])
                col_below = f"{prefix}_sig_below"
                col_above = f"{prefix}_sig_above"
                
                if col_below in row and col_above in row:
                    if not pd.isna(row[col_below]) and row[col_below]:
                        bw = bar.get_width() - 0.05
                        bc = bar.get_y() + bar.get_height() + 0.1
                        ax.text(bw, bc, '*', ha='center', va='bottom', color='black', fontweight='bold')
                    if not pd.isna(row[col_above]) and row[col_above]:
                        bw = bar.get_width() + 0.05
                        bc = bar.get_y() + bar.get_height() + 0.1
                        ax.text(bw, bc, '*', ha='center', va='bottom', color='black')
    
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel('Test')
    ax.set_xlabel('Performance Relative to Human')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1, bar_count)
    ax.invert_yaxis()
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved subtask comparison plot to {output_path}")


def main(baseline_csv: str, result_csvs: List[str], output_dir: str = './output', 
         output_format: str = 'svg', filename_prefix: str = ''):
    """Main execution function.
    
    Args:
        baseline_csv: Path to baseline CSV file containing metadata and old model scores
        result_csvs: List of paths to result CSV files (with task and raw_score columns)
        output_dir: Directory to save output figures (default: './output')
        output_format: Output format for figures - 'png', 'svg', or 'pdf' (default: 'svg')
        filename_prefix: Prefix for output filenames (default: '')
    """
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("VLM Benchmark Result Figure Generator")
    print("="*60)
    print(f"\nBaseline: {baseline_csv}")
    print(f"Results: {', '.join(result_csvs)}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {output_format}")
    
    # Load data
    print("\nLoading data...")
    baseline_data = load_baseline_data(baseline_csv)
    result_data = load_result_data(result_csvs)
    
    if not result_data:
        print("Error: No valid result files found.")
        sys.exit(1)
    
    # Merge data
    print("Merging datasets...")
    data = merge_data(baseline_data, result_data)
    
    # Prepare data
    print("Preparing data...")
    data = prepare_data(data)
    
    # Identify model columns for old and new
    old_model_cols = [col for col in data.columns if col.startswith('old_') and col.endswith('_human')]
    new_model_cols = [col for col in data.columns if col.startswith('new_') and col.endswith('_human')]
    
    # Generate plots for old models
    if old_model_cols:
        print("\n" + "="*60)
        print("ANALYSIS: OLD MODELS")
        print("="*60)
        
        # Permutation test
        perm_results_old = permutation_test(data, old_model_cols)
        
        # Stage comparison plot
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        stage_plot_path = output_dir_path / f"{prefix}stage_comparison_old.{output_format}"
        plot_stage_comparison(data, old_model_cols, str(stage_plot_path), perm_results_old)
        
        # Subtask comparison plot
        subtask_plot_path = output_dir_path / f"{prefix}subtask_comparison_old.{output_format}"
        plot_subtask_comparison(data, old_model_cols, str(subtask_plot_path))
    
    # Generate plots for new models
    if new_model_cols:
        print("\n" + "="*60)
        print("ANALYSIS: NEW MODELS")
        print("="*60)
        
        # Permutation test
        perm_results_new = permutation_test(data, new_model_cols)
        
        # Stage comparison plot
        prefix = f"{filename_prefix}_" if filename_prefix else ""
        stage_plot_path = output_dir_path / f"{prefix}stage_comparison_new.{output_format}"
        plot_stage_comparison(data, new_model_cols, str(stage_plot_path), perm_results_new)
        
        # Subtask comparison plot
        subtask_plot_path = output_dir_path / f"{prefix}subtask_comparison_new.{output_format}"
        plot_subtask_comparison(data, new_model_cols, str(subtask_plot_path))
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    # Specify your CSV files here
    
    BASELINE_CSV = "replication_with_pipeline/all_test_summary_main_open_source.csv"
    
    # Insert your result CSVs here
    RESULT_CSVS = [
        "your_result_csv_1.csv",
        "your_result_csv_2.csv",
        "your_result_csv_3.csv",
    ]
    
    OUTPUT_DIR = "./output"  # Directory to save figures
    OUTPUT_FORMAT = "svg"    # Options: 'png', 'svg', 'pdf'
    FILENAME_PREFIX = ""     # Optional prefix for output files
    
    
    
    main(
        baseline_csv=BASELINE_CSV,
        result_csvs=RESULT_CSVS,
        output_dir=OUTPUT_DIR,
        output_format=OUTPUT_FORMAT,
        filename_prefix=FILENAME_PREFIX
    )
