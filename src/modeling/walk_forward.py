"""
Walk-Forward Validation Module
Implements expanding-window Walk-Forward cross-validation with purging and embargo
to eliminate look-ahead bias and autocorrelation leakage in financial time series.
"""

from typing import List, Tuple, Generator
import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """
    Time-series expanding window splitter with embargo buffer.
    Ensures that test sets strictly succeed training sets chronologically
    with an embargo window to eliminate overlap in multi-period target labels.
    """

    def __init__(self, n_splits: int = 5, embargo_periods: int = 6, min_train_pct: float = 0.4):
        self.n_splits = n_splits
        self.embargo_periods = embargo_periods
        self.min_train_pct = min_train_pct

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) for each walk-forward fold.
        """
        n_samples = len(df)
        min_train_samples = int(n_samples * self.min_train_pct)
        test_samples_total = n_samples - min_train_samples
        test_chunk_size = test_samples_total // self.n_splits

        for fold in range(self.n_splits):
            test_start = min_train_samples + fold * test_chunk_size
            test_end = test_start + test_chunk_size if fold < self.n_splits - 1 else n_samples

            # Train ends before test_start minus the embargo window to purge overlapping targets
            train_end = max(0, test_start - self.embargo_periods)

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            yield train_indices, test_indices
