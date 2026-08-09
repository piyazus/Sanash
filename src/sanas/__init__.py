"""Sanas occupancy-estimation library.

Backend-agnostic pieces live here so the same code runs locally, on Colab or
on a rented GPU box. The Kaggle kernels under notebooks/ are self-contained
single files (Kaggle pushes one file per kernel), so ziprange.py is vendored
into notebooks/extract_subset/extract_subset.py. This package is the canonical
copy; change it here first.
"""

__all__ = ["config", "corn", "data", "labels", "models", "ziprange"]
