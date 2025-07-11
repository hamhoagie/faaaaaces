"""
Database models and initialization for FAAAAACES
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple

DATABASE_PATH = 'faaaaaces.db'

def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    
    # Videos table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            source_url TEXT,
            source_type TEXT NOT NULL CHECK(source_type IN ('upload', 'url')),
            duration_seconds REAL,
            fps REAL,
            width INTEGER,
            height INTEGER,
            file_size_bytes INTEGER,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP
        )
    ''')
    
    # Faces table - stores individual face detections
    conn.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            face_cluster_id INTEGER,
            frame_timestamp REAL NOT NULL,
            face_image_path TEXT NOT NULL,
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_width INTEGER NOT NULL,
            bbox_height INTEGER NOT NULL,
            confidence REAL NOT NULL,
            embedding TEXT NOT NULL,  -- JSON array of face embedding
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
        )
    ''')
    
    # Face clusters table - groups similar faces
    conn.execute('''
        CREATE TABLE IF NOT EXISTS face_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            representative_face_id INTEGER,
            face_count INTEGER DEFAULT 0,
            name TEXT,  -- User can assign names to clusters
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (representative_face_id) REFERENCES faces (id)
        )
    ''')
    
    # Processing jobs table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS processing_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_type TEXT NOT NULL CHECK(job_type IN ('video_processing', 'face_clustering')),
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'failed')),
            video_id INTEGER,
            progress_percent REAL DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
        )
    ''')
    
    # Create indexes for better performance
    conn.execute('CREATE INDEX IF NOT EXISTS idx_faces_video_id ON faces(video_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(face_cluster_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status)')
    
    conn.commit()
    conn.close()

class VideoModel:
    @staticmethod
    def create(filename: str, source_url: str = None, source_type: str = 'upload', **kwargs) -> int:
        """Create a new video record"""
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO videos (filename, source_url, source_type, duration_seconds, fps, width, height, file_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename, source_url, source_type,
            kwargs.get('duration_seconds'),
            kwargs.get('fps'),
            kwargs.get('width'),
            kwargs.get('height'),
            kwargs.get('file_size_bytes')
        ))
        video_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return video_id
    
    @staticmethod
    def get_by_id(video_id: int) -> Optional[Dict]:
        """Get video by ID"""
        conn = get_db_connection()
        row = conn.execute('SELECT * FROM videos WHERE id = ?', (video_id,)).fetchone()
        conn.close()
        return dict(row) if row else None
    
    @staticmethod
    def get_all() -> List[Dict]:
        """Get all videos"""
        conn = get_db_connection()
        rows = conn.execute('SELECT * FROM videos ORDER BY created_at DESC').fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def update_status(video_id: int, status: str, error_message: str = None):
        """Update video processing status"""
        conn = get_db_connection()
        if status == 'completed':
            conn.execute('''
                UPDATE videos SET status = ?, error_message = ?, processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, error_message, video_id))
        else:
            conn.execute('''
                UPDATE videos SET status = ?, error_message = ?
                WHERE id = ?
            ''', (status, error_message, video_id))
        conn.commit()
        conn.close()
    
    @staticmethod
    def delete(video_id: int):
        """Delete a video record"""
        conn = get_db_connection()
        conn.execute('DELETE FROM videos WHERE id = ?', (video_id,))
        conn.commit()
        conn.close()

class FaceModel:
    @staticmethod
    def create(video_id: int, frame_timestamp: float, face_image_path: str, 
               bbox: Tuple[int, int, int, int], confidence: float, embedding: List[float],
               mask_info: Dict = None, reconstruction_info: Dict = None) -> int:
        """Create a new face record with optional mask detection and reconstruction data"""
        conn = get_db_connection()
        
        # Prepare mask detection data
        is_masked = mask_info.get('is_masked', False) if mask_info else False
        mask_confidence = mask_info.get('confidence', 0.0) if mask_info else 0.0
        mask_type = mask_info.get('mask_type') if mask_info else None
        mask_detection_method = mask_info.get('method') if mask_info else None
        
        # Prepare reconstruction data
        has_reconstruction = reconstruction_info.get('success', False) if reconstruction_info else False
        reconstructed_image_path = reconstruction_info.get('reconstructed_image_path') if reconstruction_info else None
        reconstruction_quality = reconstruction_info.get('quality_score', 0.0) if reconstruction_info else 0.0
        reconstruction_method = reconstruction_info.get('method') if reconstruction_info else None
        
        cursor = conn.execute('''
            INSERT INTO faces (video_id, frame_timestamp, face_image_path, bbox_x, bbox_y, 
                              bbox_width, bbox_height, confidence, embedding, is_masked, mask_confidence,
                              mask_type, mask_detection_method, has_reconstruction,
                              reconstructed_image_path, reconstruction_quality, reconstruction_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id, frame_timestamp, face_image_path,
            bbox[0], bbox[1], bbox[2], bbox[3],
            confidence, json.dumps(embedding), is_masked, mask_confidence,
            mask_type, mask_detection_method, has_reconstruction,
            reconstructed_image_path, reconstruction_quality, reconstruction_method
        ))
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return face_id
    
    @staticmethod
    def get_by_video(video_id: int) -> List[Dict]:
        """Get all faces for a video"""
        conn = get_db_connection()
        rows = conn.execute('''
            SELECT f.*, fc.name as cluster_name 
            FROM faces f 
            LEFT JOIN face_clusters fc ON f.face_cluster_id = fc.id
            WHERE f.video_id = ? 
            ORDER BY f.frame_timestamp
        ''', (video_id,)).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def get_unclustered() -> List[Dict]:
        """Get faces that haven't been clustered yet"""
        conn = get_db_connection()
        rows = conn.execute('''
            SELECT * FROM faces WHERE face_cluster_id IS NULL
        ''').fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def update_cluster(face_id: int, cluster_id: int):
        """Assign face to a cluster"""
        conn = get_db_connection()
        conn.execute('UPDATE faces SET face_cluster_id = ? WHERE id = ?', (cluster_id, face_id))
        conn.commit()
        conn.close()
    
    @staticmethod
    def delete_by_video(video_id: int):
        """Delete all faces associated with a video"""
        conn = get_db_connection()
        conn.execute('DELETE FROM faces WHERE video_id = ?', (video_id,))
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_total_count() -> int:
        """Get total number of faces across all videos"""
        conn = get_db_connection()
        result = conn.execute('SELECT COUNT(*) FROM faces').fetchone()
        conn.close()
        return result[0]
    
    @staticmethod
    def get_masked_faces(video_id: int = None) -> List[Dict]:
        """Get all masked faces, optionally filtered by video"""
        conn = get_db_connection()
        if video_id:
            query = '''
                SELECT f.*, v.filename as video_filename 
                FROM faces f 
                JOIN videos v ON f.video_id = v.id 
                WHERE f.is_masked = 1 AND f.video_id = ?
                ORDER BY f.frame_timestamp
            '''
            rows = conn.execute(query, (video_id,)).fetchall()
        else:
            query = '''
                SELECT f.*, v.filename as video_filename 
                FROM faces f 
                JOIN videos v ON f.video_id = v.id 
                WHERE f.is_masked = 1
                ORDER BY f.video_id, f.frame_timestamp
            '''
            rows = conn.execute(query).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def get_reconstructed_faces(video_id: int = None) -> List[Dict]:
        """Get all faces with reconstructions, optionally filtered by video"""
        conn = get_db_connection()
        if video_id:
            query = '''
                SELECT f.*, v.filename as video_filename 
                FROM faces f 
                JOIN videos v ON f.video_id = v.id 
                WHERE f.has_reconstruction = 1 AND f.video_id = ?
                ORDER BY f.frame_timestamp
            '''
            rows = conn.execute(query, (video_id,)).fetchall()
        else:
            query = '''
                SELECT f.*, v.filename as video_filename 
                FROM faces f 
                JOIN videos v ON f.video_id = v.id 
                WHERE f.has_reconstruction = 1
                ORDER BY f.video_id, f.frame_timestamp
            '''
            rows = conn.execute(query).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def get_mask_statistics() -> Dict:
        """Get statistics about masked faces across all videos"""
        conn = get_db_connection()
        
        # Total face counts
        total_faces = conn.execute('SELECT COUNT(*) FROM faces').fetchone()[0]
        masked_faces = conn.execute('SELECT COUNT(*) FROM faces WHERE is_masked = 1').fetchone()[0]
        reconstructed_faces = conn.execute('SELECT COUNT(*) FROM faces WHERE has_reconstruction = 1').fetchone()[0]
        
        # Mask types
        mask_types = conn.execute('''
            SELECT mask_type, COUNT(*) as count 
            FROM faces 
            WHERE is_masked = 1 AND mask_type IS NOT NULL 
            GROUP BY mask_type
        ''').fetchall()
        
        # Reconstruction methods
        recon_methods = conn.execute('''
            SELECT reconstruction_method, COUNT(*) as count, AVG(reconstruction_quality) as avg_quality
            FROM faces 
            WHERE has_reconstruction = 1 AND reconstruction_method IS NOT NULL 
            GROUP BY reconstruction_method
        ''').fetchall()
        
        conn.close()
        
        return {
            'total_faces': total_faces,
            'masked_faces': masked_faces,
            'unmasked_faces': total_faces - masked_faces,
            'reconstructed_faces': reconstructed_faces,
            'mask_percentage': (masked_faces / total_faces * 100) if total_faces > 0 else 0,
            'reconstruction_percentage': (reconstructed_faces / masked_faces * 100) if masked_faces > 0 else 0,
            'mask_types': {row[0]: row[1] for row in mask_types},
            'reconstruction_methods': {row[0]: {'count': row[1], 'avg_quality': row[2]} for row in recon_methods}
        }

class FaceClusterModel:
    @staticmethod
    def create(representative_face_id: int = None, name: str = None) -> int:
        """Create a new face cluster"""
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO face_clusters (representative_face_id, name, face_count)
            VALUES (?, ?, 0)
        ''', (representative_face_id, name))
        cluster_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return cluster_id
    
    @staticmethod
    def get_all_with_faces() -> List[Dict]:
        """Get all clusters with their faces"""
        conn = get_db_connection()
        rows = conn.execute('''
            SELECT fc.*, f.face_image_path as representative_image_path,
                   COUNT(faces.id) as actual_face_count
            FROM face_clusters fc 
            LEFT JOIN faces f ON fc.representative_face_id = f.id
            LEFT JOIN faces ON fc.id = faces.face_cluster_id
            GROUP BY fc.id
            ORDER BY actual_face_count DESC
        ''').fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def update_face_count(cluster_id: int):
        """Update the face count for a cluster"""
        conn = get_db_connection()
        conn.execute('''
            UPDATE face_clusters 
            SET face_count = (SELECT COUNT(*) FROM faces WHERE face_cluster_id = ?),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (cluster_id, cluster_id))
        conn.commit()
        conn.close()

class ProcessingJobModel:
    @staticmethod
    def create(job_type: str, video_id: int = None) -> int:
        """Create a new processing job"""
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO processing_jobs (job_type, video_id)
            VALUES (?, ?)
        ''', (job_type, video_id))
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return job_id
    
    @staticmethod
    def update_status(job_id: int, status: str, progress: float = None, error_message: str = None):
        """Update job status"""
        conn = get_db_connection()
        if status == 'running' and progress is not None:
            conn.execute('''
                UPDATE processing_jobs 
                SET status = ?, progress_percent = ?, started_at = COALESCE(started_at, CURRENT_TIMESTAMP)
                WHERE id = ?
            ''', (status, progress, job_id))
        elif status in ['completed', 'failed']:
            conn.execute('''
                UPDATE processing_jobs 
                SET status = ?, progress_percent = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, progress or (100 if status == 'completed' else 0), error_message, job_id))
        else:
            conn.execute('''
                UPDATE processing_jobs 
                SET status = ?, error_message = ?
                WHERE id = ?
            ''', (status, error_message, job_id))
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_active_jobs() -> List[Dict]:
        """Get all pending or running jobs"""
        conn = get_db_connection()
        rows = conn.execute('''
            SELECT * FROM processing_jobs 
            WHERE status IN ('pending', 'running')
            ORDER BY created_at
        ''').fetchall()
        conn.close()
        return [dict(row) for row in rows]