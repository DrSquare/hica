"""
Clustering Module for Clio Privacy-Preserving Classification
Uses all-mpnet-base-v2 embeddings and k-means clustering
Based on arxiv.org/abs/2412.13678
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap


class SemanticClusterer:
    """
    Performs semantic clustering on extracted facets using embeddings.
    Uses all-mpnet-base-v2 as specified in the Clio paper.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the clusterer with the specified embedding model.

        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embeddings = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.umap_embeddings = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of shape (n_texts, embedding_dim)
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better clustering
        )
        return embeddings

    def cluster_embeddings(self,
                          embeddings: np.ndarray,
                          n_clusters: int = None,
                          min_clusters: int = 5,
                          max_clusters: int = 50,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings using k-means.

        Args:
            embeddings: NumPy array of embeddings
            n_clusters: Number of clusters (if None, will auto-determine)
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            random_state: Random state for reproducibility

        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        if n_clusters is None:
            print("Auto-determining optimal number of clusters...")
            n_clusters = self._find_optimal_clusters(
                embeddings,
                min_clusters,
                max_clusters,
                random_state
            )

        print(f"Performing k-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_

        return cluster_labels, cluster_centers

    def _find_optimal_clusters(self,
                               embeddings: np.ndarray,
                               min_clusters: int,
                               max_clusters: int,
                               random_state: int) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            embeddings: NumPy array of embeddings
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            random_state: Random state for reproducibility

        Returns:
            Optimal number of clusters
        """
        n_samples = len(embeddings)
        max_clusters = min(max_clusters, n_samples - 1)
        min_clusters = min(min_clusters, max_clusters)

        best_score = -1
        best_k = min_clusters

        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Calculate silhouette score
            score = silhouette_score(embeddings, labels)

            if score > best_score:
                best_score = score
                best_k = k

        print(f"Optimal number of clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k

    def create_umap_visualization(self,
                                 embeddings: np.ndarray,
                                 n_neighbors: int = 15,
                                 min_dist: float = 0.1,
                                 random_state: int = 42) -> np.ndarray:
        """
        Create 2D UMAP projection for visualization.

        Args:
            embeddings: NumPy array of embeddings
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random state for reproducibility

        Returns:
            NumPy array of 2D coordinates
        """
        print("Creating UMAP visualization...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='cosine'
        )
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings

    def fit_cluster(self,
                   texts: List[str],
                   n_clusters: int = None,
                   min_clusters: int = 5,
                   max_clusters: int = 50,
                   create_umap: bool = True,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Complete clustering pipeline: embed, cluster, and optionally create UMAP.

        Args:
            texts: List of text strings to cluster
            n_clusters: Number of clusters (if None, will auto-determine)
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            create_umap: Whether to create UMAP visualization
            random_state: Random state for reproducibility

        Returns:
            Dictionary containing embeddings, labels, centers, and optionally UMAP coordinates
        """
        # Generate embeddings
        self.embeddings = self.embed_texts(texts)

        # Cluster embeddings
        self.cluster_labels, self.cluster_centers = self.cluster_embeddings(
            self.embeddings,
            n_clusters,
            min_clusters,
            max_clusters,
            random_state
        )

        # Create UMAP visualization if requested
        if create_umap:
            self.umap_embeddings = self.create_umap_visualization(
                self.embeddings,
                random_state=random_state
            )

        # Organize results by cluster
        clusters = {}
        for i, label in enumerate(self.cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'index': i,
                'text': texts[i],
                'embedding': self.embeddings[i]
            })

        result = {
            'embeddings': self.embeddings,
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'clusters': clusters,
            'n_clusters': len(np.unique(self.cluster_labels))
        }

        if create_umap:
            result['umap_embeddings'] = self.umap_embeddings

        print(f"\nClustering complete:")
        print(f"  Total items: {len(texts)}")
        print(f"  Number of clusters: {result['n_clusters']}")
        print(f"  Cluster sizes: {[len(clusters[i]) for i in sorted(clusters.keys())]}")

        return result


if __name__ == "__main__":
    # Example usage
    clusterer = SemanticClusterer()

    # Sample facet descriptions (task facet from different conversations)
    sample_tasks = [
        "The task is to write Python code for sorting algorithms",
        "The task is to implement a sorting function in Python",
        "The task is to explain quantum mechanics concepts",
        "The task is to describe quantum physics principles",
        "The task is to translate English text to Spanish",
        "The task is to translate French text to German",
        "The task is to write a blog post about cooking",
        "The task is to create content about recipes and food",
        "The task is to debug JavaScript code",
        "The task is to fix errors in TypeScript code"
    ]

    print("Clustering sample tasks...")
    results = clusterer.fit_cluster(
        sample_tasks,
        n_clusters=4,
        create_umap=True
    )

    print("\n" + "="*60)
    print("CLUSTER ASSIGNMENTS:")
    print("="*60)
    for cluster_id in sorted(results['clusters'].keys()):
        items = results['clusters'][cluster_id]
        print(f"\nCluster {cluster_id} ({len(items)} items):")
        for item in items:
            print(f"  - {item['text']}")
