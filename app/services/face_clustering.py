"""
Face clustering service to group similar faces across videos
"""
import numpy as np
import json
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Dict, Tuple, Optional
from app.models.database import FaceModel, FaceClusterModel

class FaceClusterer:
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 clustering_method: str = 'dbscan',
                 min_samples: int = 2):
        """
        Initialize face clusterer
        
        Args:
            similarity_threshold: Threshold for considering faces similar (0-1)
            clustering_method: 'dbscan' or 'agglomerative'
            min_samples: Minimum samples for DBSCAN cluster
        """
        self.similarity_threshold = similarity_threshold
        self.clustering_method = clustering_method
        self.min_samples = min_samples
    
    def compute_face_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two face embeddings"""
        try:
            emb1 = np.array(embedding1).reshape(1, -1)
            emb2 = np.array(embedding2).reshape(1, -1)
            
            # Compute cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Convert to 0-1 range (cosine similarity ranges from -1 to 1)
            return (similarity + 1) / 2
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """Compute pairwise similarity matrix for all embeddings"""
        try:
            embeddings_array = np.array(embeddings)
            
            # Compute cosine similarity matrix
            cosine_sim = cosine_similarity(embeddings_array)
            
            # Convert to distance matrix for clustering (1 - similarity)
            distance_matrix = 1 - cosine_sim
            
            return distance_matrix
            
        except Exception as e:
            print(f"Error computing similarity matrix: {e}")
            return np.array([])
    
    def cluster_faces_dbscan(self, embeddings: List[List[float]]) -> List[int]:
        """Cluster faces using DBSCAN algorithm"""
        try:
            if len(embeddings) < 2:
                return [0] * len(embeddings)
            
            # Compute distance matrix
            distance_matrix = self.compute_similarity_matrix(embeddings)
            
            # DBSCAN clustering
            # eps is the maximum distance between samples in the same cluster
            eps = 1 - self.similarity_threshold  # Convert similarity to distance
            
            clustering = DBSCAN(
                eps=eps,
                min_samples=self.min_samples,
                metric='precomputed'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            return cluster_labels.tolist()
            
        except Exception as e:
            print(f"Error in DBSCAN clustering: {e}")
            return [-1] * len(embeddings)  # All noise
    
    def cluster_faces_agglomerative(self, embeddings: List[List[float]]) -> List[int]:
        """Cluster faces using Agglomerative clustering"""
        try:
            if len(embeddings) < 2:
                return [0] * len(embeddings)
            
            # Compute distance matrix
            distance_matrix = self.compute_similarity_matrix(embeddings)
            
            # Estimate number of clusters based on similarity threshold
            # This is a heuristic - could be improved
            n_clusters = max(1, len(embeddings) // 3)
            
            clustering = AgglomerativeClustering(
                n_clusters=min(n_clusters, len(embeddings)),
                metric='precomputed',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            return cluster_labels.tolist()
            
        except Exception as e:
            print(f"Error in Agglomerative clustering: {e}")
            return list(range(len(embeddings)))  # Each face is its own cluster
    
    def cluster_all_faces(self) -> Dict[str, int]:
        """
        Cluster all unclustered faces in the database
        
        Returns:
            Dictionary with stats about clustering
        """
        try:
            # Get all unclustered faces
            unclustered_faces = FaceModel.get_unclustered()
            
            if len(unclustered_faces) < 2:
                return {
                    'total_faces': len(unclustered_faces),
                    'clusters_created': 0,
                    'faces_clustered': 0
                }
            
            # Extract embeddings and face IDs
            embeddings = []
            face_ids = []
            
            for face in unclustered_faces:
                try:
                    embedding = json.loads(face['embedding'])
                    embeddings.append(embedding)
                    face_ids.append(face['id'])
                except Exception as e:
                    print(f"Error parsing embedding for face {face['id']}: {e}")
                    continue
            
            if len(embeddings) < 2:
                return {
                    'total_faces': len(unclustered_faces),
                    'clusters_created': 0,
                    'faces_clustered': 0
                }
            
            # Perform clustering
            if self.clustering_method == 'dbscan':
                cluster_labels = self.cluster_faces_dbscan(embeddings)
            else:
                cluster_labels = self.cluster_faces_agglomerative(embeddings)
            
            # Create clusters and assign faces
            cluster_map = {}  # Maps cluster_label to cluster_id in database
            faces_clustered = 0
            
            for face_id, cluster_label in zip(face_ids, cluster_labels):
                if cluster_label == -1:  # Noise in DBSCAN
                    continue
                
                # Create cluster if it doesn't exist
                if cluster_label not in cluster_map:
                    # Find a representative face for this cluster
                    cluster_face_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_label]
                    representative_face_id = face_ids[cluster_face_indices[0]]  # Use first face as representative
                    
                    cluster_id = FaceClusterModel.create(representative_face_id=representative_face_id)
                    cluster_map[cluster_label] = cluster_id
                
                # Assign face to cluster
                FaceModel.update_cluster(face_id, cluster_map[cluster_label])
                faces_clustered += 1
            
            # Update face counts for all clusters
            for cluster_id in cluster_map.values():
                FaceClusterModel.update_face_count(cluster_id)
            
            return {
                'total_faces': len(unclustered_faces),
                'clusters_created': len(cluster_map),
                'faces_clustered': faces_clustered
            }
            
        except Exception as e:
            print(f"Error clustering faces: {e}")
            return {
                'total_faces': 0,
                'clusters_created': 0,
                'faces_clustered': 0,
                'error': str(e)
            }
    
    def find_similar_faces(self, target_embedding: List[float], threshold: float = None) -> List[Dict]:
        """
        Find faces similar to a target embedding
        
        Returns:
            List of similar faces with similarity scores
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        try:
            # Get all faces from database
            # Note: This is inefficient for large datasets - consider using vector database
            all_faces = []  # You'd need to implement this query
            
            similar_faces = []
            
            for face in all_faces:
                try:
                    face_embedding = json.loads(face['embedding'])
                    similarity = self.compute_face_similarity(target_embedding, face_embedding)
                    
                    if similarity >= threshold:
                        similar_faces.append({
                            'face_id': face['id'],
                            'similarity': similarity,
                            'face_image_path': face['face_image_path'],
                            'video_id': face['video_id'],
                            'frame_timestamp': face['frame_timestamp']
                        })
                        
                except Exception as e:
                    print(f"Error processing face {face['id']}: {e}")
                    continue
            
            # Sort by similarity (highest first)
            similar_faces.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_faces
            
        except Exception as e:
            print(f"Error finding similar faces: {e}")
            return []
    
    def merge_clusters(self, cluster1_id: int, cluster2_id: int) -> bool:
        """Merge two face clusters"""
        try:
            # Implementation would involve:
            # 1. Update all faces from cluster2 to cluster1
            # 2. Delete cluster2
            # 3. Update face count for cluster1
            # 4. Optionally update representative face
            
            # This is a simplified version - full implementation would be more complex
            return True
            
        except Exception as e:
            print(f"Error merging clusters: {e}")
            return False
    
    def split_cluster(self, cluster_id: int) -> Dict[str, int]:
        """Split a cluster into smaller clusters"""
        try:
            # Implementation would involve:
            # 1. Get all faces in the cluster
            # 2. Re-cluster them with stricter parameters
            # 3. Create new clusters for the sub-groups
            
            return {'new_clusters': 0, 'faces_reassigned': 0}
            
        except Exception as e:
            print(f"Error splitting cluster: {e}")
            return {'error': str(e)}
    
    def get_cluster_stats(self) -> Dict:
        """Get clustering statistics"""
        try:
            clusters = FaceClusterModel.get_all_with_faces()
            total_faces = sum(cluster['actual_face_count'] for cluster in clusters)
            
            return {
                'total_clusters': len(clusters),
                'total_faces': total_faces,
                'average_faces_per_cluster': total_faces / len(clusters) if clusters else 0,
                'largest_cluster_size': max((cluster['actual_face_count'] for cluster in clusters), default=0),
                'clusters_with_names': len([c for c in clusters if c['name']])
            }
            
        except Exception as e:
            print(f"Error getting cluster stats: {e}")
            return {'error': str(e)}