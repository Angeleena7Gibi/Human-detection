from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt")  # Lightweight model ideal for Raspberry Pi

# Open the default camera (0 for USB or Pi Camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame (optional: improves speed on Pi)
        frame = cv2.resize(frame, (640, 480))

        # Run YOLOv8 prediction (only for 'person' class -> class 0 in COCO)
        results = model.predict(frame, conf=0.5, classes=[0], verbose=False)

        # Extract detected bounding boxes
        boxes = results[0].boxes
        human_count = len(boxes)

        # Annotate frame with detections
        annotated_frame = results[0].plot()

        # Display human count on frame
        cv2.putText(annotated_frame, f"Human Count: {human_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show result in window
        cv2.imshow("AI Human Detection", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
