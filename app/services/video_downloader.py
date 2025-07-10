"""
Video downloading service using yt-dlp with support for multiple platforms
including YouTube, Vimeo, Instagram, TikTok, and direct video links
"""
import yt_dlp
import os
import tempfile
import re
from typing import Dict, Optional, Tuple, List
from urllib.parse import urlparse
import requests

class VideoDownloader:
    def __init__(self, output_dir: str = "temp", max_size_mb: int = 1000):
        self.output_dir = output_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        os.makedirs(output_dir, exist_ok=True)
        
        # Platform-specific configurations
        self.platform_configs = {
            'instagram': {
                'format': 'best[height<=720]',  # Instagram videos are typically smaller
                'writeinfojson': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'extract_flat': False,
            },
            'tiktok': {
                'format': 'best[height<=1080]',  # TikTok videos can be up to 1080p
                'writeinfojson': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'extract_flat': False,
            },
            'youtube': {
                'format': f'best[filesize<{self.max_size_bytes}]',
                'writeinfojson': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            },
            'default': {
                'format': f'best[filesize<{self.max_size_bytes}]',
                'writeinfojson': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
        }
    
    def detect_platform(self, url: str) -> str:
        """Detect the platform from URL"""
        url_lower = url.lower()
        
        if 'instagram.com' in url_lower:
            return 'instagram'
        elif 'tiktok.com' in url_lower:
            return 'tiktok'
        elif any(domain in url_lower for domain in ['youtube.com', 'youtu.be']):
            return 'youtube'
        elif 'vimeo.com' in url_lower:
            return 'default'  # Vimeo works well with default settings
        else:
            return 'default'
    
    def get_platform_config(self, platform: str) -> Dict:
        """Get platform-specific configuration"""
        return self.platform_configs.get(platform, self.platform_configs['default'])
    
    def is_instagram_url(self, url: str) -> bool:
        """Check if URL is an Instagram URL"""
        instagram_patterns = [
            r'instagram\.com/p/',
            r'instagram\.com/reel/',
            r'instagram\.com/tv/',
            r'instagram\.com/stories/',
        ]
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in instagram_patterns)
    
    def is_tiktok_url(self, url: str) -> bool:
        """Check if URL is a TikTok URL"""
        tiktok_patterns = [
            r'tiktok\.com/@[^/]+/video/',
            r'tiktok\.com/t/',
            r'vm\.tiktok\.com/',
            r'tiktok\.com/.*?/video/',
        ]
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in tiktok_patterns)
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported platforms"""
        return [
            'YouTube (youtube.com, youtu.be)',
            'Instagram (instagram.com/p/, /reel/, /tv/)',
            'TikTok (tiktok.com, vm.tiktok.com)',
            'Vimeo (vimeo.com)',
            'Direct video URLs (.mp4, .avi, etc.)',
            'Many other sites supported by yt-dlp'
        ]
    
    def download_from_url(self, url: str) -> Tuple[bool, Optional[str], Optional[Dict], Optional[str]]:
        """
        Download video from URL using yt-dlp with platform-specific optimizations
        
        Returns:
            (success, local_file_path, video_info, error_message)
        """
        try:
            # Detect platform and get appropriate configuration
            platform = self.detect_platform(url)
            platform_config = self.get_platform_config(platform)
            
            # Configure yt-dlp options with platform-specific settings
            ydl_opts = {
                'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
                'no_warnings': True,
                'extractaudio': False,
                'writedescription': False,
                **platform_config  # Merge platform-specific config
            }
            
            # Add platform-specific headers for Instagram and TikTok
            if platform == 'instagram':
                ydl_opts.update({
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
                    }
                })
            elif platform == 'tiktok':
                ydl_opts.update({
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36'
                    }
                })
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First, extract info without downloading
                try:
                    info = ydl.extract_info(url, download=False)
                except Exception as e:
                    error_msg = self._get_platform_specific_error(platform, str(e))
                    return False, None, None, error_msg
                
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
                        'source_url': url,
                        'platform': platform,
                        'uploader': info.get('uploader'),
                        'description': info.get('description', '')[:500] if info.get('description') else None,
                        'upload_date': info.get('upload_date'),
                        'view_count': info.get('view_count'),
                        'like_count': info.get('like_count')
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
    
    def _get_platform_specific_error(self, platform: str, error_msg: str) -> str:
        """Get platform-specific error message with helpful tips"""
        error_lower = error_msg.lower()
        
        if platform == 'instagram':
            if 'private' in error_lower or 'login' in error_lower:
                return "Instagram: This account or post is private. Only public Instagram content can be processed."
            elif 'not found' in error_lower or '404' in error_lower:
                return "Instagram: Post not found. Please check the URL and ensure the content still exists."
            elif 'rate limit' in error_lower or 'too many requests' in error_lower:
                return "Instagram: Rate limited. Please wait a few minutes before trying again."
            else:
                return f"Instagram: {error_msg}. Note: Instagram frequently changes their API, try updating yt-dlp."
                
        elif platform == 'tiktok':
            if 'private' in error_lower or 'login' in error_lower:
                return "TikTok: This account is private or requires login. Only public TikTok content can be processed."
            elif 'not found' in error_lower or '404' in error_lower:
                return "TikTok: Video not found. Please check the URL and ensure the content still exists."
            elif 'region' in error_lower or 'country' in error_lower:
                return "TikTok: This content may be region-locked or unavailable in your location."
            else:
                return f"TikTok: {error_msg}. Note: Try using the direct video URL or vm.tiktok.com link."
                
        elif platform == 'youtube':
            if 'private' in error_lower:
                return "YouTube: This video is private and cannot be accessed."
            elif 'age' in error_lower:
                return "YouTube: This video is age-restricted and cannot be downloaded without authentication."
            elif 'copyright' in error_lower:
                return "YouTube: This video is protected by copyright and cannot be downloaded."
            else:
                return f"YouTube: {error_msg}"
                
        else:
            return f"Failed to extract video info: {error_msg}"
    
    def get_platform_tips(self, platform: str) -> List[str]:
        """Get platform-specific tips for users"""
        tips = {
            'instagram': [
                "Make sure the Instagram post/reel is public",
                "Use the full post URL (instagram.com/p/... or instagram.com/reel/...)",
                "Stories may not be downloadable after 24 hours",
                "Try refreshing the page if you get a 'not found' error"
            ],
            'tiktok': [
                "Make sure the TikTok video is public",
                "Use the full video URL (tiktok.com/@username/video/...)",
                "Short links (vm.tiktok.com) also work",
                "Some region-locked content may not be accessible"
            ],
            'youtube': [
                "Make sure the video is public and not age-restricted",
                "Private or unlisted videos cannot be downloaded",
                "Live streams may not be supported",
                "Very long videos may exceed size limits"
            ]
        }
        return tips.get(platform, ["Ensure the content is publicly accessible", "Check that the URL is correct"])
    
    def validate_url_format(self, url: str) -> Tuple[bool, str, str]:
        """
        Validate URL format and detect platform
        
        Returns:
            (is_valid, platform, error_message)
        """
        if not url or not url.strip():
            return False, 'unknown', "URL cannot be empty"
        
        url = url.strip()
        
        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return False, 'unknown', "URL must start with http:// or https://"
        
        platform = self.detect_platform(url)
        
        # Platform-specific validation
        if platform == 'instagram':
            if not self.is_instagram_url(url):
                return False, platform, "Please use a valid Instagram post, reel, or TV URL"
        elif platform == 'tiktok':
            if not self.is_tiktok_url(url):
                return False, platform, "Please use a valid TikTok video URL"
        
        return True, platform, ""