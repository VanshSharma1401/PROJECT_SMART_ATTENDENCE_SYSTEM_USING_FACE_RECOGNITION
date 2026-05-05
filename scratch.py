import cv2
import numpy as np
import face_recognition

def test():
    # Simulate a frame from Mac webcam
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Simulate utils.py processing
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    if len(small_frame.shape) == 2:
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)
    elif small_frame.shape[2] == 4:
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
    rgb_small = np.ascontiguousarray(rgb, dtype=np.uint8)
    
    print("Dtype:", rgb_small.dtype)
    print("Shape:", rgb_small.shape)
    print("Flags:", rgb_small.flags)
    
    try:
        face_recognition.face_locations(rgb_small, model="hog")
        print("Success")
    except Exception as e:
        print("Error:", e)

test()
