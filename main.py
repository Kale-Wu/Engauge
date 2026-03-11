import cv2
from deepface import DeepFace
from playwright.sync_api import sync_playwright
from collections import deque
import time

# Constants
ENTER_THRESHOLD = 0.45   # must exceed this to trigger boredom
EXIT_THRESHOLD = 0.20    # must be below this to trigger interest 
WINDOW = 15  # frames (~0.5s at 30fps)
FPS = 30
WEIGHT_DECAY = 1.2  # exponential weight factor (higher = more weight to recent frames)
BOREDOM_MIN_DURATION = 5  # seconds required to trigger boredom
ENGAGEMENT_MIN_DURATION = 3  # seconds required to exit boredom

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

# Duration tracking for state transitions
frames_above_threshold = 0  # frames boredom score has been above ENTER_THRESHOLD
frames_below_threshold = 0  # frames boredom score has been below EXIT_THRESHOLD
bored_start_time = None  # timestamp when score exceeded ENTER_THRESHOLD
engaged_start_time = None  # timestamp when score fell below EXIT_THRESHOLD

# Frame counter for skipping analysis
frame_count = 0

# Rolling window for smoothing boredom score
history = deque(maxlen=WINDOW)

# Main loop for video processing
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

# Release resources
cap.release()
cv2.destroyAllWindows()
