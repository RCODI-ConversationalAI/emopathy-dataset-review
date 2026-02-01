"""
Extract top 5 modalities per dataset, then top 5 papers per modality.

For each dataset:
1. Rank modalities by best score (using field-specific metric)
2. Select top 5 modalities
3. For each selected modality, get top 5 papers

Metric by modality:
- Physio, A, V → Accuracy
- T, T+A, T+V, A+V, T+A+V → F1-weighted

Excludes: SemEval, BRIGHTER, WASSA, Memotion (shared tasks - analyzed separately)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Configuration
INPUT_FILE = "EMOTION/ALL_EMOTION_METRICS_WITH_MODALITY.csv"
OUTPUT_DIR = Path("EMOTION")
TOP_N_MODALITIES = 5  # Top 5 modalities per dataset
TOP_N_PAPERS = 5      # Top 5 papers per modality

# Datasets to exclude (shared tasks - analyzed separately)
EXCLUDED_PATTERNS = [
    r'semeval',
    r'brighter',
    r'wassa',
    r'memotion',
    r'shared.?task',
]

# Modality to Metric mapping
MODALITY_METRIC_MAP = {
    'Physio': 'acc',
    'A': 'acc',
    'V': 'acc',
    'T': 'f1_weighted',
    'T+A': 'f1_weighted',
    'T+V': 'f1_weighted',
    'A+V': 'f1_weighted',
    'T+A+V': 'f1_weighted',
}

FALLBACK_METRIC = 'acc'


def should_exclude(dataset):
    """Check if dataset should be excluded (shared tasks)."""
    if pd.isna(dataset):
        return True
    dataset_lower = str(dataset).lower()
    for pattern in EXCLUDED_PATTERNS:
        if re.search(pattern, dataset_lower):
            return True
    return False


def normalize_dataset(dataset):
    """Normalize dataset names for grouping."""
    if pd.isna(dataset):
        return None

    original = str(dataset).strip()
    dataset_lower = original.lower()

    # Major datasets normalization
    if 'iemocap' in dataset_lower:
        return 'IEMOCAP'
    if 'cmu-mosei' in dataset_lower or 'cmumosei' in dataset_lower or dataset_lower == 'mosei':
        return 'CMU-MOSEI'
    if 'cmu-mosi' in dataset_lower or 'cmumosi' in dataset_lower or dataset_lower == 'mosi':
        return 'CMU-MOSI'
    if 'meld' in dataset_lower and 'fair' not in dataset_lower:
        return 'MELD'
    if 'dailydialog' in dataset_lower or 'daily_dialog' in dataset_lower:
        return 'DailyDialog'
    if 'emorynlp' in dataset_lower:
        return 'EmoryNLP'
    if 'deap' in dataset_lower and 'dreamer' not in dataset_lower:
        return 'DEAP'
    if 'dreamer' in dataset_lower:
        return 'DREAMER'
    if 'mahnob' in dataset_lower:
        return 'MAHNOB-HCI'
    if 'amigos' in dataset_lower:
        return 'AMIGOS'
    if 'wesad' in dataset_lower:
        return 'WESAD'
    if 'seed' in dataset_lower and 'iv' not in dataset_lower:
        return 'SEED'
    if 'seed-iv' in dataset_lower or 'seediv' in dataset_lower:
        return 'SEED-IV'
    if 'ravdess' in dataset_lower:
        return 'RAVDESS'
    if 'savee' in dataset_lower:
        return 'SAVEE'
    if 'crema-d' in dataset_lower or 'cremad' in dataset_lower:
        return 'CREMA-D'
    if 'tess' in dataset_lower and 'contest' not in dataset_lower:
        return 'TESS'
    if 'emo-db' in dataset_lower or 'emodb' in dataset_lower:
        return 'EMO-DB'
    if 'shemo' in dataset_lower:
        return 'ShEMO'
    if 'msp-improv' in dataset_lower:
        return 'MSP-IMPROV'
    if 'aff-wild' in dataset_lower or 'affwild' in dataset_lower:
        return 'AFF-WILD2'
    if 'fer2013' in dataset_lower:
        return 'FER2013'
    if 'ck+' in dataset_lower:
        return 'CK+'
    if 'raf-db' in dataset_lower or 'rafdb' in dataset_lower:
        return 'RAF-DB'
    if 'affectnet' in dataset_lower:
        return 'AffectNet'
    if 'afew' in dataset_lower:
        return 'AFEW'
    if 'goemotion' in dataset_lower:
        return 'GoEmotions'
    if 'emocontext' in dataset_lower:
        return 'EmoContext'
    if 'isear' in dataset_lower:
        return 'ISEAR'
    if 'iest' in dataset_lower:
        return 'IEST'
    if 'sst-2' in dataset_lower or 'sst2' in dataset_lower:
        return 'SST-2'
    if 'sst-5' in dataset_lower or 'sst5' in dataset_lower:
        return 'SST-5'
    if 'sst' in dataset_lower:
        return 'SST'
    if 'enterface' in dataset_lower:
        return 'eNTERFACE'
    if 'ch-sims' in dataset_lower:
        return 'CH-SIMS'
    if 'twitter' in dataset_lower:
        return 'Twitter'
    if 'daic-woz' in dataset_lower:
        return 'DAIC-WOZ'
    if 'bp4d' in dataset_lower:
        return 'BP4D'
    if 'physf' in dataset_lower:
        return 'PhysF'

    return original


def normalize_modality_type(modality_type):
    """Normalize modality type for grouping."""
    if pd.isna(modality_type):
        return 'unknown'

    modality_type = str(modality_type).lower().strip()

    if 'physio' in modality_type:
        return 'Physio'
    elif modality_type == 'text_only':
        return 'T'
    elif modality_type == 'audio_only':
        return 'A'
    elif modality_type == 'video_only' or modality_type == 'face_only':
        return 'V'
    elif modality_type in ['text+audio', 'audio+text']:
        return 'T+A'
    elif modality_type in ['text+video', 'video+text', 'text+face']:
        return 'T+V'
    elif modality_type in ['audio+video', 'video+audio', 'audio+face']:
        return 'A+V'
    elif 'text+audio+video' in modality_type or 'text+audio+face' in modality_type:
        return 'T+A+V'
    elif 'with_physio' in modality_type:
        return 'Physio+'
    else:
        return modality_type


def get_metric_for_modality(modality):
    """Get the appropriate metric column for a modality."""
    return MODALITY_METRIC_MAP.get(modality, FALLBACK_METRIC)


def main():
    print("=" * 70)
    print("Top 5 Modalities per Dataset, Top 5 Papers per Modality")
    print("=" * 70)
    print("\nMetric by Modality:")
    print("  Physio, A, V → Accuracy")
    print("  T, T+A, T+V, A+V, T+A+V → F1-weighted")
    print("=" * 70)

    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} rows from {INPUT_FILE}")

    # Exclude shared tasks
    df['excluded'] = df['dataset'].apply(should_exclude)
    excluded_count = df['excluded'].sum()
    df = df[~df['excluded']].copy()
    print(f"Excluded {excluded_count} shared task entries")
    print(f"Remaining: {len(df)} rows")

    # Normalize dataset and modality
    df['dataset_normalized'] = df['dataset'].apply(normalize_dataset)
    df['modality_normalized'] = df['modality_type'].apply(normalize_modality_type)

    # Filter out rows without valid dataset
    df = df[df['dataset_normalized'].notna()]
    print(f"After filtering: {len(df)} rows with valid dataset")

    # Get unique datasets
    unique_datasets = sorted(df['dataset_normalized'].unique())
    print(f"\nUnique datasets: {len(unique_datasets)}")

    all_results = []
    dataset_summary = []

    for dataset in unique_datasets:
        dataset_df = df[df['dataset_normalized'] == dataset]

        # Get modalities for this dataset
        modalities = dataset_df['modality_normalized'].unique()

        # Calculate best score for each modality
        modality_scores = []
        for modality in modalities:
            mod_df = dataset_df[dataset_df['modality_normalized'] == modality]
            metric_col = get_metric_for_modality(modality)

            # Get valid scores
            valid_scores = mod_df[metric_col].dropna()
            if len(valid_scores) == 0:
                continue

            best_score = valid_scores.max()
            paper_count = len(valid_scores)

            modality_scores.append({
                'modality': modality,
                'metric': metric_col,
                'best_score': best_score,
                'paper_count': paper_count
            })

        if not modality_scores:
            continue

        # Sort by best score and take top 5 modalities
        modality_scores = sorted(modality_scores, key=lambda x: x['best_score'], reverse=True)
        top_modalities = modality_scores[:TOP_N_MODALITIES]

        # For each top modality, get top 5 papers
        for mod_info in top_modalities:
            modality = mod_info['modality']
            metric_col = mod_info['metric']
            metric_name = 'Accuracy' if metric_col == 'acc' else 'F1-weighted'

            mod_df = dataset_df[dataset_df['modality_normalized'] == modality]
            mod_df = mod_df[mod_df[metric_col].notna()].copy()

            if len(mod_df) == 0:
                continue

            # Sort and take top 5
            mod_df = mod_df.sort_values(metric_col, ascending=False).head(TOP_N_PAPERS)

            for rank, (_, row) in enumerate(mod_df.iterrows(), 1):
                all_results.append({
                    'dataset': dataset,
                    'modality': modality,
                    'modality_rank': top_modalities.index(mod_info) + 1,
                    'paper_rank': rank,
                    'metric': metric_name,
                    'score': row[metric_col],
                    'title': row.get('title', ''),
                    'pdf_file': row.get('pdf_file', ''),
                    'source': row.get('source', '')
                })

        # Dataset summary
        dataset_summary.append({
            'dataset': dataset,
            'total_modalities': len(modalities),
            'selected_modalities': len(top_modalities),
            'modalities': ', '.join([m['modality'] for m in top_modalities]),
            'best_scores': ', '.join([f"{m['modality']}:{m['best_score']:.3f}" for m in top_modalities])
        })

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(dataset_summary)

    # Save results
    results_file = OUTPUT_DIR / "TOP5_MODALITY_TOP5_PAPERS.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n{'='*70}")
    print(f"Saved: {results_file}")
    print(f"Total papers selected: {len(results_df)}")

    summary_file = OUTPUT_DIR / "TOP5_MODALITY_SUMMARY.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved: {summary_file}")

    # Print summary for datasets with multiple modalities
    print(f"\n{'='*70}")
    print("Datasets with Multiple Modalities (Top 5 shown)")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'#Mod':>5} {'Top Modalities':<40}")
    print("-" * 70)

    multi_mod = summary_df[summary_df['total_modalities'] >= 2].sort_values(
        'total_modalities', ascending=False
    )
    for _, row in multi_mod.head(30).iterrows():
        print(f"{row['dataset']:<20} {row['total_modalities']:>5} {row['modalities']:<40}")

    # Statistics
    print(f"\n{'='*70}")
    print("Statistics")
    print(f"{'='*70}")
    print(f"Total datasets: {len(summary_df)}")
    print(f"Datasets with 2+ modalities: {len(multi_mod)}")
    print(f"Total papers selected: {len(results_df)}")

    # Papers by modality
    if len(results_df) > 0:
        mod_counts = results_df.groupby('modality').size().sort_values(ascending=False)
        print(f"\nPapers by modality:")
        for mod, cnt in mod_counts.items():
            print(f"  {mod}: {cnt}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
