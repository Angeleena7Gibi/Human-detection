# Human-detection
human detection using YOLOv8 Nano and OpenCV for real-time people counting on webcams.
<br>
Author - Angeleena gibi
#  Real-Time Human Detection using YOLOv8 and OpenCV

This project implements a real-time human detection system using the lightweight YOLOv8 Nano model with OpenCV. It captures video from a webcam or Raspberry Pi camera, detects people (COCO class 0), and displays annotated bounding boxes along with the total count of persons in each frame.

---

##  Features

-  Real-time video stream processing
-  Person-only detection using YOLOv8 Nano (`yolov8n.pt`)
-  Annotated output with bounding boxes and count


---

##  Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics (`ultralytics`)

Install dependencies:

```bash
pip install -r requirements.txt
