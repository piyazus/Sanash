"""
ReID Module
===========

Person re-identification for cross-camera tracking.
"""

from .reid_model import ReIDModel, extract_features, get_reid_model, FeatureCache
from .feature_matcher import FeatureMatcher, cosine_similarity
from .reid_service import ReIDService

__all__ = [
    'ReIDModel',
    'ReIDService', 
    'FeatureMatcher',
    'FeatureCache',
    'extract_features',
    'get_reid_model',
    'cosine_similarity',
]
