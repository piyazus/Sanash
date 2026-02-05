"""
Bus Vision - Feature Matcher
=============================

Cosine similarity matching for person re-identification.
Uses FAISS for fast similarity search when dealing with many features.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS for accelerated similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not available. Using NumPy for similarity search.")


# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================

def cosine_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.
    
    Args:
        features1: First feature vector (512,)
        features2: Second feature vector (512,)
        
    Returns:
        Similarity score between 0.0 and 1.0
        (Higher = more similar)
    """
    # Ensure normalized
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    features1 = features1 / norm1
    features2 = features2 / norm2
    
    similarity = np.dot(features1, features2)
    
    # Clamp to [0, 1] (can be slightly > 1 due to float errors)
    return float(np.clip(similarity, 0.0, 1.0))


def cosine_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute cosine distance (1 - similarity).
    
    Returns:
        Distance between 0.0 and 1.0
        (Lower = more similar)
    """
    return 1.0 - cosine_similarity(features1, features2)


def euclidean_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute Euclidean distance between feature vectors.
    """
    return float(np.linalg.norm(features1 - features2))


def batch_cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and all gallery features.
    
    Args:
        query: Query feature (512,) or (N, 512)
        gallery: Gallery features (M, 512)
        
    Returns:
        Similarity scores (M,) or (N, M)
    """
    # Normalize
    query = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-6)
    gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-6)
    
    # Compute similarity
    if query.ndim == 1:
        return np.dot(gallery, query)
    else:
        return np.dot(query, gallery.T)


# =============================================================================
# FEATURE MATCHER (NumPy)
# =============================================================================

class FeatureMatcher:
    """
    Match person features across cameras using similarity search.
    
    Maintains a gallery of known person features and finds matches
    for new query features.
    
    Usage:
        matcher = FeatureMatcher(threshold=0.7)
        matcher.add_feature(track_id=1, features=feat1)
        matcher.add_feature(track_id=2, features=feat2)
        
        match_id, score = matcher.find_match(query_features)
        if match_id is not None:
            print(f"Matched to track {match_id} with score {score}")
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        feature_dim: int = 512,
        use_faiss: bool = True
    ):
        """
        Initialize feature matcher.
        
        Args:
            threshold: Minimum similarity to consider a match (0.0 - 1.0)
            feature_dim: Dimension of feature vectors
            use_faiss: Use FAISS for fast search if available
        """
        self.threshold = threshold
        self.feature_dim = feature_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # Storage
        self.features: List[np.ndarray] = []
        self.track_ids: List[int] = []
        self.metadata: List[dict] = []
        
        # FAISS index
        self.index = None
        if self.use_faiss:
            self._init_faiss_index()
    
    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search."""
        # Use inner product index (same as cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.feature_dim)
        logger.debug("FAISS index initialized")
    
    def add_feature(
        self,
        track_id: int,
        features: np.ndarray,
        metadata: Optional[dict] = None
    ):
        """
        Add features to the gallery.
        
        Args:
            track_id: Unique identifier for this person
            features: Feature vector (512,)
            metadata: Optional metadata (camera_id, timestamp, etc.)
        """
        # Normalize features
        features = features.astype(np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        self.features.append(features)
        self.track_ids.append(track_id)
        self.metadata.append(metadata or {})
        
        if self.use_faiss:
            self.index.add(features.reshape(1, -1))
    
    def find_match(
        self,
        query: np.ndarray,
        top_k: int = 1,
        exclude_ids: Optional[List[int]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Find best matching track for query features.
        
        Args:
            query: Query feature vector (512,)
            top_k: Number of top matches to consider
            exclude_ids: Track IDs to exclude from matching
            
        Returns:
            (track_id, similarity_score) or (None, 0.0) if no match
        """
        if len(self.features) == 0:
            return None, 0.0
        
        # Normalize query
        query = query.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        if self.use_faiss:
            return self._find_match_faiss(query, top_k, exclude_ids)
        else:
            return self._find_match_numpy(query, top_k, exclude_ids)
    
    def _find_match_numpy(
        self,
        query: np.ndarray,
        top_k: int,
        exclude_ids: Optional[List[int]]
    ) -> Tuple[Optional[int], float]:
        """Find match using NumPy."""
        gallery = np.array(self.features)
        similarities = np.dot(gallery, query)
        
        # Apply exclusions
        if exclude_ids:
            for i, tid in enumerate(self.track_ids):
                if tid in exclude_ids:
                    similarities[i] = -1
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.threshold:
            return self.track_ids[best_idx], float(best_score)
        
        return None, float(best_score)
    
    def _find_match_faiss(
        self,
        query: np.ndarray,
        top_k: int,
        exclude_ids: Optional[List[int]]
    ) -> Tuple[Optional[int], float]:
        """Find match using FAISS (faster for large galleries)."""
        k = min(top_k + len(exclude_ids or []), len(self.features))
        
        scores, indices = self.index.search(query.reshape(1, -1), k)
        scores = scores[0]
        indices = indices[0]
        
        for i, idx in enumerate(indices):
            if idx == -1:
                continue
            track_id = self.track_ids[idx]
            score = scores[i]
            
            if exclude_ids and track_id in exclude_ids:
                continue
            
            if score >= self.threshold:
                return track_id, float(score)
        
        return None, float(scores[0]) if len(scores) > 0 else 0.0
    
    def find_matches_batch(
        self,
        queries: np.ndarray,
        top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        Find matches for multiple queries at once.
        
        Args:
            queries: Query features (N, 512)
            top_k: Number of top matches per query
            
        Returns:
            List of lists: [[(track_id, score), ...], ...]
        """
        if len(self.features) == 0:
            return [[] for _ in range(len(queries))]
        
        # Normalize queries
        queries = queries.astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / (norms + 1e-6)
        
        results = []
        
        if self.use_faiss:
            k = min(top_k, len(self.features))
            scores, indices = self.index.search(queries, k)
            
            for i in range(len(queries)):
                matches = []
                for j in range(k):
                    idx = indices[i, j]
                    if idx != -1 and scores[i, j] >= self.threshold:
                        matches.append((self.track_ids[idx], float(scores[i, j])))
                results.append(matches)
        else:
            gallery = np.array(self.features)
            all_sims = np.dot(queries, gallery.T)
            
            for i in range(len(queries)):
                sims = all_sims[i]
                top_indices = np.argsort(sims)[::-1][:top_k]
                matches = [
                    (self.track_ids[idx], float(sims[idx]))
                    for idx in top_indices
                    if sims[idx] >= self.threshold
                ]
                results.append(matches)
        
        return results
    
    def update_feature(self, track_id: int, features: np.ndarray, alpha: float = 0.9):
        """
        Update existing features with exponential moving average.
        
        New_features = alpha * old_features + (1 - alpha) * new_features
        
        Args:
            track_id: Track to update
            features: New features
            alpha: Weight for old features (0.9 = mostly keep old)
        """
        # Normalize new features
        features = features.astype(np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        # Find track
        for i, tid in enumerate(self.track_ids):
            if tid == track_id:
                # EMA update
                old_features = self.features[i]
                updated = alpha * old_features + (1 - alpha) * features
                # Re-normalize
                updated = updated / (np.linalg.norm(updated) + 1e-6)
                self.features[i] = updated
                
                # Rebuild FAISS index (required after update)
                if self.use_faiss:
                    self._rebuild_index()
                
                return True
        
        return False
    
    def _rebuild_index(self):
        """Rebuild FAISS index after modifications."""
        if self.use_faiss and len(self.features) > 0:
            self.index = faiss.IndexFlatIP(self.feature_dim)
            gallery = np.array(self.features)
            self.index.add(gallery)
    
    def remove_track(self, track_id: int):
        """Remove a track from the gallery."""
        for i, tid in enumerate(self.track_ids):
            if tid == track_id:
                del self.features[i]
                del self.track_ids[i]
                del self.metadata[i]
                
                if self.use_faiss:
                    self._rebuild_index()
                
                return True
        
        return False
    
    def get_track_features(self, track_id: int) -> Optional[np.ndarray]:
        """Get features for a specific track."""
        for i, tid in enumerate(self.track_ids):
            if tid == track_id:
                return self.features[i]
        return None
    
    def clear(self):
        """Clear all stored features."""
        self.features = []
        self.track_ids = []
        self.metadata = []
        
        if self.use_faiss:
            self._init_faiss_index()
    
    @property
    def size(self) -> int:
        """Number of tracks in gallery."""
        return len(self.track_ids)
    
    def __len__(self) -> int:
        return self.size


# =============================================================================
# TEMPORAL MATCHER
# =============================================================================

class TemporalFeatureMatcher(FeatureMatcher):
    """
    Feature matcher with temporal awareness.
    
    Only matches features that appeared within a time window.
    Useful for cross-camera matching where transitions happen quickly.
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        time_window: float = 5.0,
        feature_dim: int = 512
    ):
        """
        Args:
            time_window: Maximum seconds between appearances to consider matching
        """
        super().__init__(threshold=threshold, feature_dim=feature_dim)
        self.time_window = time_window
        self.timestamps: List[float] = []
    
    def add_feature(
        self,
        track_id: int,
        features: np.ndarray,
        timestamp: float,
        metadata: Optional[dict] = None
    ):
        """Add features with timestamp."""
        super().add_feature(track_id, features, metadata)
        self.timestamps.append(timestamp)
    
    def find_match(
        self,
        query: np.ndarray,
        query_time: float,
        top_k: int = 1,
        exclude_ids: Optional[List[int]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Find match within time window.
        """
        if len(self.features) == 0:
            return None, 0.0
        
        # Find valid time indices
        valid_indices = [
            i for i, t in enumerate(self.timestamps)
            if abs(query_time - t) <= self.time_window
        ]
        
        if not valid_indices:
            return None, 0.0
        
        # Get valid features
        valid_features = [self.features[i] for i in valid_indices]
        valid_track_ids = [self.track_ids[i] for i in valid_indices]
        
        # Normalize query
        query = query.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Find best match
        gallery = np.array(valid_features)
        similarities = np.dot(gallery, query)
        
        # Apply exclusions
        if exclude_ids:
            for i, tid in enumerate(valid_track_ids):
                if tid in exclude_ids:
                    similarities[i] = -1
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.threshold:
            return valid_track_ids[best_idx], float(best_score)
        
        return None, float(best_score)
    
    def cleanup_old(self, current_time: float, max_age: float = 60.0):
        """Remove features older than max_age seconds."""
        cutoff = current_time - max_age
        
        indices_to_keep = [
            i for i, t in enumerate(self.timestamps)
            if t >= cutoff
        ]
        
        self.features = [self.features[i] for i in indices_to_keep]
        self.track_ids = [self.track_ids[i] for i in indices_to_keep]
        self.metadata = [self.metadata[i] for i in indices_to_keep]
        self.timestamps = [self.timestamps[i] for i in indices_to_keep]
        
        if self.use_faiss:
            self._rebuild_index()


if __name__ == "__main__":
    # Test feature matcher
    print("Testing Feature Matcher...")
    
    matcher = FeatureMatcher(threshold=0.7)
    
    # Create test features
    feat1 = np.random.randn(512).astype(np.float32)
    feat2 = feat1 + np.random.randn(512).astype(np.float32) * 0.1  # Similar
    feat3 = np.random.randn(512).astype(np.float32)  # Different
    
    # Add to gallery
    matcher.add_feature(track_id=1, features=feat1)
    matcher.add_feature(track_id=2, features=feat3)
    
    # Query similar feature
    match_id, score = matcher.find_match(feat2)
    print(f"Query similar: matched={match_id}, score={score:.3f}")
    assert match_id == 1, "Should match track 1"
    
    # Query different feature
    different = np.random.randn(512).astype(np.float32)
    match_id, score = matcher.find_match(different)
    print(f"Query different: matched={match_id}, score={score:.3f}")
    
    print(f"Gallery size: {len(matcher)}")
    print("Feature Matcher test passed!")
