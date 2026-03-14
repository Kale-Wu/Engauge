import cv2
from deepface import DeepFace
from collections import deque
import time
import threading
import subprocess
import signal
import os
import tempfile
import shutil

# Constants
ENTER_THRESHOLD = 0.45   # must exceed this to trigger boredom
EXIT_THRESHOLD = 0.20    # must be below this to trigger interest 
WINDOW = 4  # frames (~0.5s at 30fps)
FPS = 30
WEIGHT_DECAY = 1.2  # exponential weight factor (higher = more weight to recent frames)
BOREDOM_MIN_DURATION = 5  # seconds required to trigger boredom
ENGAGEMENT_MIN_DURATION = 3  # seconds required to exit boredom

# Apps to display (left to right)
APPS = [
    {"name": "youtube", "url": "https://www.youtube.com/shorts"},
    {"name": "instagram", "url": "https://www.instagram.com/reels/"},
    {"name": "tiktok", "url": "https://www.tiktok.com/@for_you"},
    {"name": "linkedin", "url": "https://www.linkedin.com/feed/"},
]

NUM_APPS = len(APPS)
APP_WIDTH_FRACTION = 1.0 / NUM_APPS  # Each app gets equal width

# Initialize video capture (0 is the default webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Calculate delay in milliseconds for the FPS rate
frame_delay = int(1000 / FPS)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open video feed")
    exit()

print("Video feed started. Press 'q' to quit.")

# Engagement state tracking (starts as engaged)
engagement_state = "Engaged"
previous_engagement_state = "Engaged"  # Track previous state for transitions

# Browser process management
app_processes = {}  # {app_name: subprocess.Popen}
app_user_dirs = {}  # {app_name: temp_dir_path}
app_management_active = True
app_management_lock = threading.Lock()

# Duration tracking for state transitions
frames_above_threshold = 0  # frames boredom score has been above ENTER_THRESHOLD
frames_below_threshold = 0  # frames boredom score has been below EXIT_THRESHOLD
bored_start_time = None  # timestamp when score exceeded ENTER_THRESHOLD
engaged_start_time = None  # timestamp when score fell below EXIT_THRESHOLD

# Frame counter for skipping analysis
frame_count = 0

# Rolling window for smoothing boredom score
history = deque(maxlen=WINDOW)

# App management functions using subprocess
def open_single_app(app):
    """Open a single app in its own browser process."""
    import ctypes
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    
    app_index = APPS.index(app)
    app_width = int(screen_width / NUM_APPS)
    app_x = app_width * app_index
    
    chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
    
    # Check if Chrome exists, otherwise try Chromium or Edge
    if not os.path.exists(chrome_path):
        chrome_path = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = "C:/Program Files/Microsoft/Edge/Application/msedge.exe"
    
    try:
        with app_management_lock:
            # Don't open if already open
            if app["name"] in app_processes:
                return
            
            # Create isolated user-data-dir for this app instance
            user_dir = tempfile.mkdtemp(prefix=f"chrome_{app['name']}_")
            app_user_dirs[app["name"]] = user_dir
            
            proc = subprocess.Popen([
                chrome_path,
                f"--app={app['url']}",
                f"--window-position={app_x},0",
                f"--window-size={app_width},{screen_height}",
                f"--user-data-dir={user_dir}",
                "--no-first-run",
                "--disable-extensions",
                "--incognito",
            ])
            app_processes[app["name"]] = proc
            print(f"Opened {app['name']} at x={app_x}, size={app_width}x{screen_height}. PID: {proc.pid}")
    except Exception as e:
        print(f"Error opening {app['name']}: {e}")

def close_single_app(app_name):
    """Close a single app's browser process and clean up temp profile."""
    with app_management_lock:
        proc = app_processes.pop(app_name, None)
        user_dir = app_user_dirs.pop(app_name, None)
    
    if proc:
        try:
            print(f"Killing process tree for {app_name} (PID: {proc.pid})...")
            # Use taskkill to kill entire process tree (/T flag)
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True
            )
            print(f"Killed {app_name}")
        except Exception as e:
            print(f"Error closing {app_name}: {e}")
    else:
        print(f"{app_name} not found in active processes")
    
    # Clean up temp profile directory
    if user_dir and os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir, ignore_errors=True)
            print(f"Cleaned up profile for {app_name}")
        except Exception as e:
            print(f"Error cleaning up profile for {app_name}: {e}")

def open_all_apps():
    """Open all apps."""
    for app in APPS:
        if app["name"] not in app_processes:
            open_single_app(app)

def close_all_apps():
    """Close all active app browsers."""
    # Extract list of app names while holding lock, then release lock before closing
    with app_management_lock:
        app_names_to_close = list(app_processes.keys())
    
    # Close each app outside the lock to avoid deadlock
    for app_name in app_names_to_close:
        close_single_app(app_name)

def open_apps_threaded():
    """Request to open all apps in background thread."""
    thread = threading.Thread(target=open_all_apps, daemon=True)
    thread.start()

def close_apps_threaded(wait=False):
    """Request to close all apps in background thread."""
    thread = threading.Thread(target=close_all_apps, daemon=False)
    thread.start()
    if wait:
        thread.join(timeout=10)  # Wait up to 10 seconds for closure

# Main loop for video processing
try:
    while True:
        # Read frame from the video feed
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Analyze emotions using DeepFace
        frame_count += 1
        if frame_count % 3 == 0:
            try:
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'],
                    detector_backend="retinaface",  # More accurate face detection
                    enforce_detection=False
                    )
                
                # Get the dominant emotion and all emotion scores from the result
                if result and len(result) > 0:
                    dominant_emotion = result[0]['dominant_emotion']
                    emotions = result[0]['emotion']
                    
                    # Extract emotion values (default to 0 if not present)
                    neutral = emotions.get('neutral', 0)
                    sad = emotions.get('sad', 0)
                    angry = emotions.get('angry', 0)
                    disgust = emotions.get('disgust', 0)
                    
                    # Calculate boredom score (normalized to 0-1)
                    boredom_score = ((neutral * 0.5) + (sad * 0.3) + (angry * 0.1) + (disgust * 0.1)) / 100
                    
                    # Add to rolling window and calculate weighted smoothed score
                    history.append(boredom_score)
                    
                    # Calculate weighted average (more recent frames get higher weights)
                    weighted_sum = 0
                    weight_total = 0
                    for i, score in enumerate(history):
                        # Weight increases exponentially towards the end (most recent)
                        weight = WEIGHT_DECAY ** i
                        weighted_sum += score * weight
                        weight_total += weight
                    
                    smoothed_score = weighted_sum / weight_total
                    
                    # Track state transitions
                    previous_engagement_state = engagement_state
                    # Duration-based state machine logic for engagement
                    if engagement_state == "Engaged":
                        if smoothed_score > ENTER_THRESHOLD:
                            # Track when threshold is first exceeded
                            if bored_start_time is None:
                                bored_start_time = time.time()
                            
                            elapsed = time.time() - bored_start_time
                            # Transition to bored if sustained for minimum duration
                            if elapsed >= BOREDOM_MIN_DURATION:
                                engagement_state = "Bored"
                                bored_start_time = None
                                # Transition to bored - open apps
                                if previous_engagement_state != "Bored":
                                    open_apps_threaded()
                            print(f"Boredom Score: {smoothed_score:.2f} | Duration above threshold: {elapsed:.1f}s/{BOREDOM_MIN_DURATION}s | Status: {engagement_state}")
                        else:
                            bored_start_time = None
                            print(f"Boredom Score: {smoothed_score:.2f} | Status: {engagement_state}")
                    elif engagement_state == "Bored":
                        if smoothed_score < EXIT_THRESHOLD:
                            # Track when threshold is first fallen below
                            if engaged_start_time is None:
                                engaged_start_time = time.time()
                            
                            elapsed = time.time() - engaged_start_time
                            # Transition to engaged if sustained for minimum duration
                            if elapsed >= ENGAGEMENT_MIN_DURATION:
                                engagement_state = "Engaged"
                                engaged_start_time = None
                                # Transition to engaged - close apps
                                if previous_engagement_state != "Engaged":
                                    close_apps_threaded()
                            print(f"Boredom Score: {smoothed_score:.2f} | Duration below threshold: {elapsed:.1f}s/{ENGAGEMENT_MIN_DURATION}s | Status: {engagement_state}")
                        else:
                            engaged_start_time = None
                            print(f"Boredom Score: {smoothed_score:.2f} | Status: {engagement_state}")
            except Exception as e:
                print(f"Error analyzing emotion: {e}")
        
        # Display the frame in a window
        cv2.imshow('Video Feed', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user (Ctrl+C).")
    print("Cleaning up...")

finally:
    # Release resources
    print("Closing video feed...")
    cap.release()
    cv2.destroyAllWindows()
    
    # Close all apps
    print("Closing all app browsers...")
    close_all_apps()  # Call directly (blocking) instead of threaded
    time.sleep(1)  # Wait for apps to fully close
    
    print("Program ended.")
