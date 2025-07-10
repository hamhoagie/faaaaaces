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
               bbox: Tuple[int, int, int, int], confidence: float, embedding: List[float]) -> int:
        """Create a new face record"""
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO faces (video_id, frame_timestamp, face_image_path, bbox_x, bbox_y, 
                              bbox_width, bbox_height, confidence, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id, frame_timestamp, face_image_path,
            bbox[0], bbox[1], bbox[2], bbox[3],
            confidence, json.dumps(embedding)
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