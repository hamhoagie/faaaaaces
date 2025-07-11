#!/usr/bin/env python3
"""
Database upgrade script to add mask detection and reconstruction fields
"""
import sqlite3
import os
import sys

# Add parent directory to path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def upgrade_database():
    """Add mask detection and reconstruction fields to faces table"""
    
    db_path = "faaaaaces.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    print("🔄 Upgrading database for mask detection and reconstruction...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if mask detection columns already exist
        cursor.execute("PRAGMA table_info(faces)")
        columns = [row[1] for row in cursor.fetchall()]
        
        new_columns = []
        
        # Add mask detection columns if they don't exist
        if 'is_masked' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN is_masked BOOLEAN DEFAULT 0")
            new_columns.append('is_masked')
        
        if 'mask_confidence' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN mask_confidence REAL DEFAULT 0.0")
            new_columns.append('mask_confidence')
        
        if 'mask_type' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN mask_type TEXT DEFAULT NULL")
            new_columns.append('mask_type')
        
        if 'mask_detection_method' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN mask_detection_method TEXT DEFAULT NULL")
            new_columns.append('mask_detection_method')
        
        # Add reconstruction columns if they don't exist
        if 'has_reconstruction' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN has_reconstruction BOOLEAN DEFAULT 0")
            new_columns.append('has_reconstruction')
        
        if 'reconstructed_image_path' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN reconstructed_image_path TEXT DEFAULT NULL")
            new_columns.append('reconstructed_image_path')
        
        if 'reconstruction_quality' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN reconstruction_quality REAL DEFAULT 0.0")
            new_columns.append('reconstruction_quality')
        
        if 'reconstruction_method' not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN reconstruction_method TEXT DEFAULT NULL")
            new_columns.append('reconstruction_method')
        
        conn.commit()
        conn.close()
        
        if new_columns:
            print(f"✅ Added columns: {', '.join(new_columns)}")
        else:
            print("✅ Database already up to date")
        
        return True
        
    except Exception as e:
        print(f"❌ Error upgrading database: {e}")
        return False

if __name__ == "__main__":
    success = upgrade_database()
    sys.exit(0 if success else 1)