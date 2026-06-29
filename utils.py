import re
import requests
from typing import Tuple, Optional, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, InvalidVideoId

def extract_video_id(url: str) -> str:
    """Extracts the YouTube video ID from various formats of YouTube URLs."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return ""

def fetch_youtube_metadata(video_id: str) -> Dict[str, Any]:
    """Fetches video metadata using the YouTube oEmbed API."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "title": data.get("title", "Unknown Title"),
                "author_name": data.get("author_name", "Unknown Channel"),
                "thumbnail_url": data.get("thumbnail_url", "")
            }
    except Exception:
        pass
    return {"title": "Unknown Title", "author_name": "Unknown Channel", "thumbnail_url": ""}

def get_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetches the transcript of a YouTube video given its ID."""
    try:
        api = YouTubeTranscriptApi()
        
        # Get the list of all available transcripts
        transcript_list = api.list(video_id)
        
        # Extract all available language codes to pass them as fallback languages
        available_languages = [t.language_code for t in transcript_list]
        
        transcript = api.fetch(video_id, languages=available_languages)
        # Combine text segments into a single string
        text = " ".join([t.text for t in transcript])
        return text, None
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcript found for this video."
    except InvalidVideoId:
        return None, "Invalid Video ID."
    except Exception as e:
        return None, f"An error occurred while fetching the transcript: {str(e)}"