import os
import json
import time
import subprocess
import tempfile
import random
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import backoff

# New dependency: VideoDB
from videodb import connect
import videodb.exceptions

# ========================================
# Configuration
# ========================================
FREESOUND_API_KEY = "sqErJpDfTQ7UdjxqQBnJgnjoAMN5RbKLNwiKJdl6"  # Replace with actual key
VIDEODB_API_KEY = "sk-aNrSlpaCUQbuKDKYb_psvIKy-JvtP21a2Kon96a0s_8"      # Replace with actual key
API_SEARCH = "https://freesound.org/apiv2/search/"
CACHE_FILE = "search_cache.pkl"
FEEDBACK_FILE = "audio_feedback.json"

# ========================================
# Logging
# ========================================
def log(msg):
    print(msg, flush=True)

def run_cmd(cmd):
    log("RUN: " + " ".join(cmd))
    subprocess.check_call(cmd)

# ========================================
# Search Cache System
# ========================================
class SearchCache:
    """Cache search results to reduce API calls"""
    def __init__(self, cache_file=CACHE_FILE, ttl_hours=24):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.ttl = timedelta(hours=ttl_hours)
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                log(f"[CACHE] Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            log(f"[CACHE] Failed to save cache: {e}")
    
    def cache_key(self, mood, tags, descriptors_str):
        """Generate consistent cache key"""
        tags_str = '.'.join(sorted(set(tags)))
        return f"{mood}_{tags_str}_{descriptors_str}"
    
    def get(self, mood, tags, descriptors_str):
        key = self.cache_key(mood, tags, descriptors_str)
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                log(f"[CACHE] Cache hit for {mood}")
                return cached_data
            else:
                del self.cache[key]
        return None
    
    def set(self, mood, tags, descriptors_str, results):
        key = self.cache_key(mood, tags, descriptors_str)
        self.cache[key] = (results, datetime.now())
        self._save_cache()

# ========================================
# Feedback System for Learning
# ========================================
class AudioFeedbackSystem:
    """Track search effectiveness for continuous improvement"""
    def __init__(self, feedback_file=FEEDBACK_FILE):
        self.feedback_file = feedback_file
        self.feedback_history = self._load_feedback()
    
    def _load_feedback(self):
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                log(f"[FEEDBACK] Failed to load feedback: {e}")
        return []
    
    def _save_feedback(self):
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            log(f"[FEEDBACK] Failed to save feedback: {e}")
    
    def log_search_attempt(self, video_id, mood, tags, queries, selected_sound, user_rating=None):
        """Log a search attempt for future analysis"""
        entry = {
            'video_id': video_id,
            'mood': mood,
            'tags': tags,
            'queries': queries,
            'sound_id': selected_sound.get('id'),
            'sound_name': selected_sound.get('name'),
            'sound_tags': selected_sound.get('tags', []),
            'sound_rating': selected_sound.get('avg_rating'),
            'user_rating': user_rating,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_history.append(entry)
        self._save_feedback()
        log(f"[FEEDBACK] Logged attempt for video {video_id}")
    
    def get_successful_patterns(self, min_rating=4.0):
        """Analyze which search patterns led to successful selections"""
        # Filter by high user ratings or high avg_rating
        successful = [
            f for f in self.feedback_history 
            if ((f.get('user_rating') or 0) >= min_rating or (f.get('sound_rating') or 0) >= min_rating)
        ]
        
        patterns = defaultdict(list)
        for entry in successful:
            mood = entry.get('mood')
            tags = tuple(sorted(entry.get('tags', [])))
            patterns[(mood, tags)].append(entry)
        
        return patterns

# ========================================
# Step 1: VideoDB Director Analysis (Simplified)
# ========================================

def describe_with_retry(obj, prompt, timeout_seconds=300, max_attempts=5):
    """
    Retry logic for VideoDB describe calls with exponential backoff.
    
    Args:
        obj: VideoDB object (scene or video)
        prompt: Description prompt
        timeout_seconds: Total timeout across all retries (default 300s = 5 min)
        max_attempts: Maximum number of attempts (default 5)
    """
    @backoff.on_exception(
        backoff.expo,
        videodb.exceptions.RequestTimeoutError,
        max_tries=max_attempts,
        max_time=timeout_seconds,
        factor=2  # Double wait time each retry: 1s, 2s, 4s, 8s, 16s...
    )
    def _describe():
        return obj.describe(prompt=prompt)
    
    try:
        return _describe()
    except videodb.exceptions.RequestTimeoutError as e:
        log(f"[WARN] Failed to describe after {max_attempts} retries (timeout: {timeout_seconds}s): {e}")
        return None
    except Exception as e:
        log(f"[WARN] Describe failed with unexpected error: {e}")
        return None

def analyze_video_with_director(video_path):
    """
    Uploads video to VideoDB and performs scene-by-scene analysis
    to determine mood and generate search queries with enhanced metadata.
    No indexing required - uses extract_scenes() directly.
    """
    log(f"[DIRECTOR] Connecting to VideoDB...")
    conn = connect(api_key=VIDEODB_API_KEY)
    
    # 1. Upload
    log(f"[DIRECTOR] Uploading {video_path}...")
    video = conn.upload(file_path=str(video_path))
    log(f"[DIRECTOR] Video ID: {video.id}")
    
    # 2. Extract scenes (no indexing needed)
    try:
        log("[DIRECTOR] Extracting video scenes for detailed analysis...")
        scene_collection = video.extract_scenes()
        scenes = scene_collection.scenes
        log(f"[DIRECTOR] Extracted {len(scenes)} scenes")
        
        if not scenes:
            log("[WARN] No scenes detected, using fallback metadata")
            return get_fallback_metadata()
        
        # Perform scene-by-scene analysis (LIMITED TO 2-3 KEY SCENES FOR SPEED)
        scene_analyses = []
        # Analyze fewer scenes: first scene + middle scene + last scene
        max_scenes_to_analyze = min(3, len(scenes))
        scenes_to_check = []
        
        if len(scenes) == 1:
            scenes_to_check = scenes[:1]
        elif len(scenes) == 2:
            scenes_to_check = scenes[:2]
        else:
            # First, middle, and last scenes
            scenes_to_check = [
                scenes[0],
                scenes[len(scenes) // 2],
                scenes[-1]
            ]
        
        for i, scene in enumerate(scenes_to_check):
            log(f"[DIRECTOR] Analyzing scene {i+1}/{len(scenes_to_check)}: {scene.start}-{scene.end}s")
            
            # SHORTER, MORE FOCUSED PROMPT for faster processing
            scene_prompt = (
                "Respond with ONLY valid JSON (no markdown). Describe this scene:\n"
                "{\n"
                '  "content": "visual elements (be specific)",\n'
                '  "mood": "one word",\n'
                '  "tags": ["tag1", "tag2"]\n'
                "}"
            )
            
            scene_result = describe_with_retry(scene, scene_prompt, timeout_seconds=120, max_attempts=4)
            
            if scene_result and len(scene_result) > 10:
                scene_analyses.append(scene_result)
                log(f"[DIRECTOR] Scene {i+1} analysis: {scene_result[:80]}...")
            else:
                log(f"[WARN] Empty response for scene {i+1}, continuing...")
        
        # 4. Synthesize all scene analyses into final metadata
        if scene_analyses:
            log(f"[DIRECTOR] Synthesizing {len(scene_analyses)} scene analyses...")
            synthesis_prompt = (
                f"Based on these scene-by-scene analyses:\n"
                f"{json.dumps(scene_analyses, indent=2)}\n\n"
                "Synthesize the overall video metadata. Return ONLY a valid JSON object:\n"
                "{\n"
                '  "mood": "overall one-word mood (cinematic, calm, energetic, dark, bright, dramatic, etc)",\n'
                '  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],  # Freesound tags like ambient, atmospheric, pad, etc\n'
                '  "search_queries": ["query1", "query2", "query3", "query4", "query5"],  # CRITICAL: First queries MUST be EXACT content-based from the scenes (e.g., "basketball court", "ocean waves", "city traffic", "rain forest"). Then add mood-based queries (e.g., "energetic sports background", "calm ambient"). Be SPECIFIC about what you see.\n'
                '  "audio_descriptors": {\n'
                '    "bpm": [min, max],  # e.g., [60, 100]\n'
                '    "brightness": [min, max],  # 0-100, e.g., [30, 70]\n'
                '    "loudness": [min, max],  # LUFS, e.g., [-20, -8]\n'
                '    "spectral_centroid": [min, max]  # Hz, e.g., [2000, 5000]\n'
                "  },\n"
                '  "duration_range": [min_seconds, max_seconds],  # e.g., [20, 300]\n'
                '  "min_rating": 3.5,  # Minimum sound rating\n'
                '  "avoid_tags": ["tag1", "tag2"]  # Tags to exclude, e.g., ["vocal", "solo-instrument"]\n'
                "}\n"
                "IMPORTANT: Prioritize SPECIFIC content-based searches first (what you literally see in the scenes).\n"
                "Do not write markdown formatting. Return only valid JSON."
            )
            
            # SYNTHESIZE with retry logic and LONGER timeouts
            log("[DIRECTOR] Synthesizing scene analyses with extended timeout (5 minutes)...")
            result = describe_with_retry(
                scenes[0],
                prompt=synthesis_prompt,
                timeout_seconds=300,  # 5 minutes for synthesis
                max_attempts=5
            )
            
            if result and len(result) > 10:  # Valid response
                log(f"[DIRECTOR] Raw Output: {result}")
                return result
            else:
                log(f"[WARN] Synthesis failed after all retries, using fallback metadata")
        
    except Exception as e:
        log(f"[WARN] Scene extraction failed: {e}, using fallback metadata")
    
    # If extraction or synthesis fails, return fallback metadata
    log("[DIRECTOR] Using fallback metadata")
    return get_fallback_metadata()

def get_fallback_metadata():
    """
    Returns default fallback metadata when AI analysis fails.
    """
    return '{"mood": "neutral", "tags": ["ambient", "background", "atmospheric"], "search_queries": ["background music", "ambient soundtrack", "cinematic background"], "audio_descriptors": {}, "duration_range": [20, 300], "min_rating": 3.0, "avoid_tags": ["vocal", "speech"]}'
# ========================================
# Step 2: Metadata Interpreter (Enhanced)
# ========================================
def parse_director_output(llm_response):
    """
    Cleans and parses the LLM text output into usable Python objects.
    Handles both old and new enhanced formats.
    """
    try:
        # Strip potential markdown code blocks if the LLM adds them
        clean_text = llm_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        mood = data.get("mood", "unknown")
        queries = data.get("search_queries", [])
        tags = data.get("tags", [])
        audio_descriptors = data.get("audio_descriptors", {})
        duration_range = data.get("duration_range", [20, 300])
        min_rating = data.get("min_rating", 3.0)
        avoid_tags = data.get("avoid_tags", [])
        
        # Fallback if no queries
        if not queries:
            queries = [f"{mood} background music", "cinematic ambient"]
        
        # Fallback if no tags
        if not tags:
            tags = ["ambient", "background", "atmospheric"]
            
        log(f"[INTERPRETER] Mood: {mood}")
        log(f"[INTERPRETER] Tags: {tags}")
        log(f"[INTERPRETER] Queries: {queries}")
        log(f"[INTERPRETER] Descriptors: {audio_descriptors}")
        
        return {
            'queries': queries,
            'mood': mood,
            'tags': tags,
            'audio_descriptors': audio_descriptors,
            'duration_range': duration_range,
            'min_rating': min_rating,
            'avoid_tags': avoid_tags
        }
        
    except json.JSONDecodeError as e:
        log(f"[WARN] Director returned invalid JSON: {e}. Falling back.")
        return {
            'queries': ["ambient cinematic background", "soft instrumental"],
            'mood': "neutral",
            'tags': ["ambient", "background"],
            'audio_descriptors': {},
            'duration_range': [20, 300],
            'min_rating': 3.0,
            'avoid_tags': []
        }

# ========================================
# Step 3: Advanced Freesound Search
# ========================================
def search_freesound(query, token, filter_str="", fields=None, sort="rating_desc", page_size=15):
    """Enhanced search with filters and custom fields"""
    if fields is None:
        fields = "id,name,tags,previews,duration,license,avg_rating,num_downloads,bpm,brightness,loudness"
    
    params = {
        "query": query,
        "fields": fields,
        "page_size": page_size,
        "sort": sort
    }
    
    if filter_str:
        params["filter"] = filter_str
    
    r = requests.get(
        API_SEARCH,
        headers={"Authorization": f"Token {token}"},
        params=params,
        timeout=30
    )
    r.raise_for_status()
    return r.json()

def search_with_tags_and_keywords(metadata, token, cache=None):
    """
    Multi-strategy search combining tags and keywords with descriptor filtering.
    Prioritizes exact content-based queries first, then falls back to mood/tags.
    Returns prioritized results.
    """
    mood = metadata.get('mood', 'ambient')
    tags = metadata.get('tags', [])
    queries = metadata.get('queries', [])
    audio_descriptors = metadata.get('audio_descriptors', {})
    duration_range = metadata.get('duration_range', [20, 300])
    min_rating = metadata.get('min_rating', 3.0)
    avoid_tags = metadata.get('avoid_tags', [])
    
    # Check cache first
    descriptors_str = json.dumps(audio_descriptors, sort_keys=True)
    if cache:
        cached = cache.get(mood, tags, descriptors_str)
        if cached:
            return cached
    
    results_pool = []
    log(f"[SEARCH] Starting multi-strategy search for mood: {mood}")
    
    # Strategy 1: PRIORITY - Exact content-based queries (e.g., "basketball court", "ocean waves")
    # These are specific to the video content and should be searched FIRST
    log("[SEARCH] Strategy 1: PRIORITY - Exact content-based queries")
    if queries:
        for query in queries:  # Search ALL queries, prioritizing exact matches
            try:
                # Try exact phrase search first with relaxed filters for maximum results
                filter_str = f"duration:[{duration_range[0]} TO {duration_range[1]}]"
                res = search_freesound(query, token, filter_str=filter_str, sort="rating_desc", page_size=20)
                results = res.get("results", [])
                log(f"[SEARCH] Exact query '{query}': Found {len(results)} results")
                
                if results:
                    # Mark these as high priority by adding them first
                    results_pool.extend(results)
                    log(f"[SEARCH] ✓ Content match found for '{query}' - prioritizing these results")
            except Exception as e:
                log(f"[WARN] Exact query search failed for '{query}': {e}")
    
    # Strategy 2: Tag-based search (only if we need more results)
    if len(results_pool) < 10:
        log("[SEARCH] Strategy 2: Tag-based filtering (supplementary)")
        for tag in tags[:3]:  # Top 3 tags
            try:
                filter_str = f"tag:{tag} duration:[{duration_range[0]} TO {duration_range[1]}] avg_rating:[{min_rating} TO *]"
                res = search_freesound("", token, filter_str=filter_str, sort="rating_desc", page_size=15)
                results = res.get("results", [])
                log(f"[SEARCH] Tag '{tag}': Found {len(results)} results")
                results_pool.extend(results)
            except Exception as e:
                log(f"[WARN] Tag search failed for '{tag}': {e}")
    
    # Strategy 3: Mood-based fallback (last resort)
    if len(results_pool) < 5:
        log("[SEARCH] Strategy 3: Mood-based fallback (last resort)")
        try:
            filter_str = f"duration:[{duration_range[0]} TO {duration_range[1]}] avg_rating:[{min_rating} TO *]"
            res = search_freesound(f"{mood} background music", token, filter_str=filter_str, sort="rating_desc", page_size=10)
            results = res.get("results", [])
            log(f"[SEARCH] Mood query: Found {len(results)} results")
            results_pool.extend(results)
        except Exception as e:
            log(f"[WARN] Mood search failed: {e}")
    
    # Remove duplicates by ID
    seen = set()
    unique_results = []
    for r in results_pool:
        if r['id'] not in seen:
            seen.add(r['id'])
            unique_results.append(r)
    
    # Filter out avoid_tags
    if avoid_tags:
        filtered = []
        for sound in unique_results:
            sound_tags = set(sound.get('tags', []))
            if not sound_tags & set(avoid_tags):
                filtered.append(sound)
        unique_results = filtered
        log(f"[SEARCH] After avoiding tags: {len(unique_results)} results remain")
    
    # Apply descriptor filters if provided
    if audio_descriptors:
        unique_results = filter_by_audio_descriptors(unique_results, audio_descriptors)
        log(f"[SEARCH] After descriptor filtering: {len(unique_results)} results")
    
    log(f"[SEARCH] Final pool: {len(unique_results)} results")
    
    # Cache results
    if cache:
        cache.set(mood, tags, descriptors_str, unique_results)
    
    return unique_results

def filter_by_audio_descriptors(candidates, target_descriptors):
    """
    Filter sounds by their audio analysis descriptors.
    """
    filtered = []
    
    for sound in candidates:
        match = True
        
        # Check BPM if specified
        if 'bpm' in target_descriptors and target_descriptors['bpm']:
            target_bpm = target_descriptors['bpm']
            sound_bpm = sound.get('bpm')
            if sound_bpm is not None:
                if not (target_bpm[0] <= sound_bpm <= target_bpm[1]):
                    match = False
        
        # Check brightness if specified
        if match and 'brightness' in target_descriptors and target_descriptors['brightness']:
            target_brightness = target_descriptors['brightness']
            sound_brightness = sound.get('brightness')
            if sound_brightness is not None:
                if not (target_brightness[0] <= sound_brightness <= target_brightness[1]):
                    match = False
        
        # Check loudness if specified
        if match and 'loudness' in target_descriptors and target_descriptors['loudness']:
            target_loudness = target_descriptors['loudness']
            sound_loudness = sound.get('loudness')
            if sound_loudness is not None:
                if not (target_loudness[0] <= sound_loudness <= target_loudness[1]):
                    match = False
        
        # Check spectral_centroid if specified
        if match and 'spectral_centroid' in target_descriptors and target_descriptors['spectral_centroid']:
            target_sc = target_descriptors['spectral_centroid']
            sound_sc = sound.get('spectral_centroid')
            if sound_sc is not None:
                if not (target_sc[0] <= sound_sc <= target_sc[1]):
                    match = False
        
        if match:
            filtered.append(sound)
    
    return filtered

def score_and_select_candidate(candidates, video_duration, mood_metadata, feedback_system=None):
    """
    Score candidates based on multiple criteria and select the best one.
    Uses machine learning insights from feedback when available.
    """
    if not candidates:
        return None
    
    scored = []
    mood = mood_metadata.get('mood', 'neutral')
    
    for sound in candidates:
        score = 0
        
        # 1. Duration compatibility (can loop, but shouldn't be too short)
        duration = sound.get('duration', 0)
        if 30 <= duration <= 180:
            score += 20
        elif 15 <= duration <= 300:
            score += 10
        
        # 2. Rating (higher is better)
        rating = sound.get('avg_rating', 0)
        score += rating * 10  # Max +50
        
        # 3. Downloads popularity
        downloads = sound.get('num_downloads', 0)
        score += min(downloads / 1000, 15)  # Max +15
        
        # 4. Tag relevance (bonus for specific tags)
        tags = sound.get('tags', [])
        relevant_tags = {'ambient', 'background', 'cinematic', 'pad', 'atmospheric', 'loop', 'theme', 'soundtrack'}
        tag_matches = len(set(tags) & relevant_tags)
        score += tag_matches * 5  # Max +40
        
        # 5. Mood-specific adjustments
        if mood == 'calm' or mood == 'peaceful':
            brightness = sound.get('brightness', 50)
            if brightness < 60:
                score += 10
        
        if mood == 'energetic' or mood == 'dramatic':
            brightness = sound.get('brightness', 50)
            if brightness >= 50:
                score += 10
        
        # 6. Learning from feedback patterns
        if feedback_system:
            patterns = feedback_system.get_successful_patterns()
            mood_tuple = (mood, tuple(sorted(set(mood_metadata.get('tags', [])))))
            if mood_tuple in patterns:
                # Check if this sound type has appeared in successful patterns
                successful_sound_ids = {p['sound_id'] for p in patterns[mood_tuple]}
                if sound['id'] in successful_sound_ids:
                    score += 25  # Strong boost for known-good sounds
        
        scored.append({
            'sound': sound,
            'score': score
        })
    
    # Sort by score and return top choice
    top_choices = sorted(scored, key=lambda x: x['score'], reverse=True)[:3]
    if top_choices:
        # Add some variety: pick from top 3 randomly weighted
        selected = random.choices(
            [tc['sound'] for tc in top_choices],
            weights=[tc['score'] for tc in top_choices],
            k=1
        )[0]
        log(f"[SELECT] Chose sound with score {[tc['score'] for tc in top_choices if tc['sound']['id'] == selected['id']][0]}")
        return selected
    
    return None

def search_by_reference_sound(reference_sound_id, token, similarity_space='laion_clap'):
    """
    Find sounds similar to a known good reference using content-based similarity.
    """
    try:
        log(f"[SIMILAR] Searching for sounds similar to reference {reference_sound_id}")
        r = requests.get(
            f"https://freesound.org/apiv2/sounds/{reference_sound_id}/similar/",
            headers={"Authorization": f"Token {token}"},
            params={
                "similarity_space": similarity_space,
                "fields": "id,name,tags,previews,duration,avg_rating,num_downloads",
                "page_size": 20
            },
            timeout=30
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        log(f"[SIMILAR] Found {len(results)} similar sounds")
        return results
    except Exception as e:
        log(f"[WARN] Similar search failed: {e}")
        return []

# ========================================
# Step 4: Video + Audio Composition (FFmpeg)
# ========================================
def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    return float(subprocess.check_output(cmd).strip())

def mix_audio_with_video(video, bg, output, bg_vol=0.3, original_vol=1.0):
    # Mixes the new background audio with the original video audio
    # The 'amix' filter handles mixing two audio streams
    # bg_vol: volume for background audio (default 0.3 to keep it subtle)
    # original_vol: volume for original video audio (default 1.0 to preserve clarity)
    
    # Check if video has audio stream first
    probe = subprocess.run(
        ["ffprobe", "-i", video, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
        stdout=subprocess.PIPE
    )
    has_original_audio = len(probe.stdout) > 0

    if has_original_audio:
        # Use dynamic range compression to prevent audio clipping and ensure balance
        fcomplex = (
            f"[0:a]volume={original_vol}[a0];"
            f"[1:a]volume={bg_vol}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=3,dynaudnorm=p=0.9:s=5[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video, "-i", bg,
            "-filter_complex", fcomplex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            output
        ]
    else:
        # If original video is silent, just add the new track with reduced volume
        cmd = [
            "ffmpeg", "-y",
            "-i", video, "-i", bg,
            "-filter_complex", f"[1:a]volume={bg_vol}[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest", # Cut audio to video length
            output
        ]

    run_cmd(cmd)

# ========================================
# Download Preview Audio from Freesound
# ========================================
def download_preview(sound, out_dir, token):
    """
    Downloads the MP3 preview for a Freesound sound object.
    Returns the path to the downloaded file, or None on failure.
    """
    try:
        previews = sound.get('previews', {})
        mp3_url = previews.get('preview-hq-mp3') or previews.get('preview-lq-mp3')
        if not mp3_url:
            log(f"[DOWNLOAD] No MP3 preview found for sound {sound.get('id')}")
            return None
        local_path = os.path.join(out_dir, f"freesound_{sound['id']}.mp3")
        log(f"[DOWNLOAD] Downloading preview: {mp3_url}")
        headers = {"Authorization": f"Token {token}"}
        r = requests.get(mp3_url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        log(f"[DOWNLOAD] Saved to {local_path}")
        return local_path
    except Exception as e:
        log(f"[DOWNLOAD] Failed to download preview: {e}")
        return None

# ========================================
# MAIN PIPELINE (Enhanced)
# ========================================
def run_pipeline(video_path, bg_volume=0.25, original_volume=2.0, output_path=None, user_rating=None, skip_ai=False):
    """
    Enhanced pipeline with intelligent search, caching, feedback, and scoring.
    
    Args:
        video_path: Path to input video
        bg_volume: Background audio volume (0.0-1.0)
        original_volume: Original video audio volume (0.0-2.0+)
        output_path: Path for output video (auto-generated if None)
        user_rating: Optional rating for feedback (1.0-5.0)
        skip_ai: If True, skip VideoDB AI analysis and use generic search (faster, no server errors)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if output_path is None:
        output_path = video_path.with_name(video_path.stem + "_director_edit.mp4")
    
    log(f"\n=== Processing {video_path} with Enhanced VideoDB Director ===")
    log(f"[AUDIO] Background volume: {bg_volume}, Original volume: {original_volume}")
    log(f"[MODE] AI Analysis: {'Disabled (using generic search)' if skip_ai else 'Enabled'}")

    # Initialize systems
    cache = SearchCache()
    feedback_system = AudioFeedbackSystem()
    tmp = tempfile.mkdtemp(prefix="director_")
    
    # --- 1. Video Analysis (VideoDB Director or Skip) ---
    log("[PIPELINE] Step 1: Video Analysis")
    
    if skip_ai:
        # Use generic metadata without AI analysis
        log("[PIPELINE] Skipping AI analysis, using generic metadata")
        metadata = {
            'queries': ["background music", "ambient soundtrack", "cinematic background", "instrumental music"],
            'mood': "neutral",
            'tags': ["ambient", "background", "atmospheric", "cinematic"],
            'audio_descriptors': {},
            'duration_range': [20, 300],
            'min_rating': 3.5,
            'avoid_tags': ["vocal", "speech", "talking"]
        }
    else:
        # Full AI analysis
        director_response_text = analyze_video_with_director(video_path)
        # --- 2. Metadata Interpreter ---
        log("[PIPELINE] Step 2: Parse Metadata")
        metadata = parse_director_output(director_response_text)
    
    # --- 3. Enhanced Audio Retrieval (Multi-strategy Search) ---
    log("[PIPELINE] Step 3: Enhanced Audio Search")
    candidates = search_with_tags_and_keywords(metadata, FREESOUND_API_KEY, cache=cache)
    
    if not candidates:
        log("[WARN] No candidates found, falling back to simple search")
        simple_res = search_freesound("ambient background music", FREESOUND_API_KEY)
        candidates = simple_res.get("results", [])
    
    # --- 4. Intelligent Selection ---
    log("[PIPELINE] Step 4: Intelligent Sound Selection")
    video_dur = get_video_duration(str(video_path))
    selected_sound = score_and_select_candidate(
        candidates, 
        video_dur, 
        metadata,
        feedback_system=feedback_system
    )
    
    if not selected_sound:
        raise RuntimeError("No suitable sound found")

    log(f"[CHOSEN] {selected_sound['name']} (id={selected_sound['id']}, rating={selected_sound.get('avg_rating', 'N/A')})")
    
    # --- 5. Download Preview ---
    log("[PIPELINE] Step 5: Download Audio")
    mp3 = download_preview(selected_sound, tmp, FREESOUND_API_KEY)
    
    if not mp3:
        raise RuntimeError("Failed to download preview audio")

    # --- 6. Composition (FFmpeg) ---
    log("[PIPELINE] Step 6: Mix Audio with Video")
    mix_audio_with_video(str(video_path), mp3, str(output_path), bg_volume, original_volume)

    # --- 7. Log Feedback ---
    log("[PIPELINE] Step 7: Logging Feedback")
    feedback_system.log_search_attempt(
        video_id=video_path.stem,
        mood=metadata.get('mood'),
        tags=metadata.get('tags'),
        queries=metadata.get('queries'),
        selected_sound=selected_sound,
        user_rating=user_rating
    )

    log(f"\n[DONE] Output: {output_path}\n")
    return str(output_path)

