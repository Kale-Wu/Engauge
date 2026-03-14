"""
Watches the webcam for signs of boredom using facial emotion analysis.
When boredom is detected and sustained, launches social media apps
tiled side-by-side. Closes them once engagement returns.
"""

import cv2
from deepface import DeepFace
from collections import deque
import time
import threading
import subprocess
import os
import tempfile
import shutil

# Two separate thresholds prevent flickering at the boundary — the score
# has to rise high to trigger boredom, but fall much lower to exit it.
ENTER_THRESHOLD = 0.45          # Score must exceed this to start boredom timer
EXIT_THRESHOLD = 0.20           # Score must fall below this to start recovery timer
BOREDOM_MIN_DURATION = 5        # Seconds above threshold before opening apps
ENGAGEMENT_MIN_DURATION = 3     # Seconds below threshold before closing apps

# Raw per-frame scores are noisy, so we average over a short rolling window.
# WEIGHT_DECAY makes recent frames count more than older ones.
WINDOW = 4
WEIGHT_DECAY = 1.2

FPS = 30
EMOTION_ANALYSIS_INTERVAL = 3   # Only analyze every 3rd frame to reduce lag

# Add or remove apps here. Each gets an equal slice of the screen.
APPS = [
    {"name": "youtube",   "url": "https://www.youtube.com/shorts"},
    {"name": "instagram", "url": "https://www.instagram.com/reels/"},
    {"name": "tiktok",    "url": "https://www.tiktok.com/@mythosmondays/video/7606796928923225366"},
]

NUM_APPS = len(APPS)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, FPS)
frame_delay = int(1000 / FPS)

if not cap.isOpened():
    print("Error: Could not open video feed")
    exit()

print("Video feed started. Press 'q' to quit.")

# Current state of the user, and timers for how long they've been in that state.
engagement_state = "Engaged"
bored_start_time = None
engaged_start_time = None
history = deque(maxlen=WINDOW)

# Tracks open browser processes and their temp profile directories.
# The lock prevents open/close calls from colliding across threads.
app_processes = {}
app_user_dirs = {}
app_management_lock = threading.Lock()


def find_chrome_path():
    """Check common install locations and return the first Chrome/Edge found."""
    candidates = [
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise RuntimeError("Chrome or Edge not found. Please install Google Chrome.")


def get_app_window_geometry(app_index):
    """Calculate position and size so all apps tile evenly across the screen."""
    import ctypes
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    app_width = screen_width // NUM_APPS
    app_x = app_width * app_index

    return (app_x, 0, app_width, screen_height)


def open_single_app(app):
    """Launch a Chrome window for one app with its own isolated profile."""
    app_index = APPS.index(app)
    app_x, _, app_width, app_height = get_app_window_geometry(app_index)
    chrome_path = find_chrome_path()

    with app_management_lock:
        if app["name"] in app_processes:
            return

        # Each app gets a throwaway temp profile so they don't share sessions.
        user_dir = tempfile.mkdtemp(prefix=f"chrome_{app['name']}_")
        app_user_dirs[app["name"]] = user_dir

        proc = subprocess.Popen([
            chrome_path,
            f"--app={app['url']}",
            f"--window-position={app_x},0",
            f"--window-size={app_width},{app_height}",
            f"--user-data-dir={user_dir}",
            "--no-first-run",
            "--disable-extensions",
            "--incognito",
        ])
        app_processes[app["name"]] = proc
        print(f"[+] {app['name']:12} PID {proc.pid:5} | x={app_x}, size={app_width}x{app_height}")


def close_single_app(app_name):
    """Kill a Chrome window and delete its temp profile."""
    with app_management_lock:
        proc = app_processes.pop(app_name, None)
        user_dir = app_user_dirs.pop(app_name, None)

    if proc:
        try:
            # Chrome spawns multiple child processes, so we kill the whole tree.
            # /F = force kill, /T = include all children of this PID.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5
            )
            print(f"[-] {app_name:12} PID {proc.pid:5} | Killed")
        except Exception as e:
            print(f"[!] {app_name:12} | Error closing: {e}")

    if user_dir and os.path.exists(user_dir):
        shutil.rmtree(user_dir, ignore_errors=True)


def open_all_apps():
    for app in APPS:
        if app["name"] not in app_processes:
            open_single_app(app)


def close_all_apps():
    with app_management_lock:
        app_names_to_close = list(app_processes.keys())
    threads = [threading.Thread(target=close_single_app, args=(name,), daemon=False) for name in app_names_to_close]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def open_apps_threaded():
    threading.Thread(target=open_all_apps, daemon=True).start()


def close_apps_threaded(wait=False):
    t = threading.Thread(target=close_all_apps, daemon=False)
    t.start()
    if wait:
        t.join(timeout=10)


def analyze_frame_emotions(frame):
    """Run DeepFace on a frame and return the emotion scores, or None on failure."""
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend="retinaface",
            enforce_detection=False
        )
        return result[0]['emotion'] if result else None
    except Exception as e:
        print(f"[!] Emotion analysis failed: {e}")
        return None


def calculate_boredom_score(emotions):
    """
    Collapse emotion percentages into a single boredom score between 0 and 1.
    Neutral is weighted heaviest since it's the most common boredom signal.
    """
    if not emotions:
        return 0.0
    return ((emotions.get('neutral', 0) * 0.5 +
             emotions.get('sad', 0) * 0.3 +
             emotions.get('angry', 0) * 0.1 +
             emotions.get('disgust', 0) * 0.1) / 100)


def update_smoothed_score(raw_score):
    """Add the latest score to the rolling window and return the weighted average."""
    history.append(raw_score)
    weighted_sum = sum(score * (WEIGHT_DECAY ** i) for i, score in enumerate(history))
    weight_total = sum(WEIGHT_DECAY ** i for i in range(len(history)))
    return weighted_sum / weight_total


def update_engagement_state(smoothed_score):
    """
    Transition between Engaged and Bored based on how long the score stays
    above or below its respective threshold. The gap between ENTER_THRESHOLD
    and EXIT_THRESHOLD prevents rapid back-and-forth at the boundary.
    """
    global engagement_state, bored_start_time, engaged_start_time

    previous_state = engagement_state

    if engagement_state == "Engaged":
        if smoothed_score > ENTER_THRESHOLD:
            if bored_start_time is None:
                bored_start_time = time.time()
            elapsed = time.time() - bored_start_time
            if elapsed >= BOREDOM_MIN_DURATION:
                engagement_state = "Bored"
                bored_start_time = None
                if previous_state != "Bored":
                    open_apps_threaded()
            print(f"[→] Score: {smoothed_score:.2f} | Boredom: {elapsed:.1f}s/{BOREDOM_MIN_DURATION}s | State: {engagement_state}")
        else:
            bored_start_time = None
            print(f"[→] Score: {smoothed_score:.2f} | State: {engagement_state}")

    elif engagement_state == "Bored":
        if smoothed_score < EXIT_THRESHOLD:
            if engaged_start_time is None:
                engaged_start_time = time.time()
            elapsed = time.time() - engaged_start_time
            if elapsed >= ENGAGEMENT_MIN_DURATION:
                engagement_state = "Engaged"
                engaged_start_time = None
                if previous_state != "Engaged":
                    close_apps_threaded()
            print(f"[←] Score: {smoothed_score:.2f} | Recovery: {elapsed:.1f}s/{ENGAGEMENT_MIN_DURATION}s | State: {engagement_state}")
        else:
            engaged_start_time = None
            print(f"[←] Score: {smoothed_score:.2f} | State: {engagement_state}")


frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        frame_count += 1
        if frame_count % EMOTION_ANALYSIS_INTERVAL == 0:
            emotions = analyze_frame_emotions(frame)
            if emotions:
                raw_score = calculate_boredom_score(emotions)
                smoothed_score = update_smoothed_score(raw_score)
                update_engagement_state(smoothed_score)

        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[*] Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    close_all_apps()
    print("[*] Done.")
