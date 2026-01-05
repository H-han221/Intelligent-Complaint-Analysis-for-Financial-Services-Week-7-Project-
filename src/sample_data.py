import pandas as pd

def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    total_samples: int = 12000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample based on product category.
    """
    samples = []

    proportions = df[label_col].value_counts(normalize=True)

    for label, proportion in proportions.items():
        n_samples = int(total_samples * proportion)
        subset = df[df[label_col] == label].sample(
            n=min(n_samples, len(df[df[label_col] == label])),
            random_state=random_state
        )
        samples.append(subset)

    return pd.concat(samples).reset_index(drop=True)
