"""
Video downloading service using yt-dlp
"""
import yt_dlp
import os
import tempfile
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import requests

class VideoDownloader:
    def __init__(self, output_dir: str = "temp", max_size_mb: int = 1000):
        self.output_dir = output_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        os.makedirs(output_dir, exist_ok=True)
    
    def download_from_url(self, url: str) -> Tuple[bool, Optional[str], Optional[Dict], Optional[str]]:
        """
        Download video from URL using yt-dlp
        
        Returns:
            (success, local_file_path, video_info, error_message)
        """
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
                'format': 'best[filesize<{}]'.format(self.max_size_bytes),
                'no_warnings': True,
                'extractaudio': False,
                'writeinfojson': False,
                'writedescription': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First, extract info without downloading
                try:
                    info = ydl.extract_info(url, download=False)
                except Exception as e:
                    return False, None, None, f"Failed to extract video info: {str(e)}"
                
                # Check file size
                if 'filesize' in info and info['filesize']:
                    if info['filesize'] > self.max_size_bytes:
                        return False, None, None, f"Video too large: {info['filesize']} bytes (max: {self.max_size_bytes})"
                
                # Download the video
                try:
                    ydl.download([url])
                except Exception as e:
                    return False, None, None, f"Download failed: {str(e)}"
                
                # Find the downloaded file
                expected_filename = ydl.prepare_filename(info)
                if os.path.exists(expected_filename):
                    video_info = {
                        'title': info.get('title', 'Unknown'),
                        'duration': info.get('duration'),
                        'width': info.get('width'),
                        'height': info.get('height'),
                        'fps': info.get('fps'),
                        'filesize': info.get('filesize') or os.path.getsize(expected_filename),
                        'format': info.get('ext'),
                        'source_url': url
                    }
                    return True, expected_filename, video_info, None
                else:
                    return False, None, None, "Downloaded file not found"
                    
        except Exception as e:
            return False, None, None, f"Unexpected error: {str(e)}"
    
    def download_direct_video(self, url: str) -> Tuple[bool, Optional[str], Optional[Dict], Optional[str]]:
        """
        Download video file directly (for direct video URLs)
        
        Returns:
            (success, local_file_path, video_info, error_message)
        """
        try:
            # Parse URL to get filename
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = f"video_{hash(url)}.mp4"
            
            local_path = os.path.join(self.output_dir, filename)
            
            # Download with streaming
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_size_bytes:
                return False, None, None, f"Video too large: {content_length} bytes"
            
            # Download the file
            with open(local_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Check size during download
                        if downloaded > self.max_size_bytes:
                            f.close()
                            os.remove(local_path)
                            return False, None, None, "Video too large during download"
            
            # Basic video info
            video_info = {
                'title': filename,
                'filesize': os.path.getsize(local_path),
                'source_url': url,
                'format': filename.split('.')[-1] if '.' in filename else 'unknown'
            }
            
            return True, local_path, video_info, None
            
        except requests.RequestException as e:
            return False, None, None, f"Download error: {str(e)}"
        except Exception as e:
            return False, None, None, f"Unexpected error: {str(e)}"
    
    def is_supported_url(self, url: str) -> bool:
        """Check if URL is supported by yt-dlp"""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                # Just try to extract info, don't download
                ydl.extract_info(url, download=False)
                return True
        except:
            return False
    
    def get_video_info(self, url: str) -> Optional[Dict]:
        """Get video information without downloading"""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration'),
                    'width': info.get('width'),
                    'height': info.get('height'),
                    'fps': info.get('fps'),
                    'filesize': info.get('filesize'),
                    'format': info.get('ext'),
                    'description': info.get('description', '')[:500],  # Truncate
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date')
                }
        except:
            return None
    
    def cleanup_file(self, file_path: str):
        """Remove downloaded file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass  # Ignore cleanup errors