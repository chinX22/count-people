import torch
import cv2
import random
import sys
import warnings

use_case = ("Use case : python3 detect.py I/V (for image or video) path/to/source"
            " (or 0 if webcam) output_name (if using video file input or "
            "if you want image output")
example1 = ("Example 1 : python3 detect.py V demo.mp4 demo_out")
example2 = ("Example 2 : python3 detect.py I demo.jpg")

# Error check
if len(sys.argv) < 3:
        print(use_case)
        sys.exit()
# load source
if sys.argv[1] == 'V':
        if sys.argv[2] == "0":
                droidcam_ip = "10.0.0.65"  # Your DroidCam IP address
                port = "4747"  # Default DroidCam port
                rtsp_url = f"http://{droidcam_ip}:{port}/video"  # RTSP URL for DroidCam
                cap = cv2.VideoCapture(rtsp_url)
        else:
                cap = cv2.VideoCapture(sys.argv[2])
        if not cap.isOpened():
                print("Couldn't open video")
                sys.exit()
elif sys.argv[1] == 'I':
        source = sys.argv[2]
        img = cv2.imread(source)
        if img is None:
                print("Couldn't open image at " + source)
                sys.exit()
else:
        print(use_case)
        print(example1)
        print(example2)
        sys.exit()


# Load pre trained yolov5s
warnings.filterwarnings("ignore", category=FutureWarning)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#pop = 0

def detect(frame):
        # Pass image through model, boxes collects coords for all directions, then
        # filter for only those with people
        output = model(frame)
        boxes = output.xyxy[0]
        peopleBoxes = boxes[boxes[:, 5] == 0]

        pop = len(peopleBoxes)

        # Draw boxes for every person found with varying colors
        for *box, in peopleBoxes:
                rand = random.randint(0, 255)
                box = list(map(int, box))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), 
                              (rand, 255 - rand, 0), 2)
                
        cv2.putText(frame, f"Found: {pop}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.3, (255, 255, 255), 2)
                
        return frame



if sys.argv[1] == 'V':
        # If using a webcam, a frame rate limiter will be applied to reduce
        # latency in output
        if sys.argv[2] == "0":
                frameCount = 0
                while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                                break
                        if (frameCount % 10) == 0:
                                frameCount += 1
                                # Show image with boxes
                                cv2.imshow("Output", detect(frame))
                                if cv2.waitKey(1) == ord('q'):
                                        break
                        else : frameCount += 1
                cap.release()
        else:
                out_name = "output" if len(sys.argv) == 3 else sys.argv[3]
                frame_size = (int(cap.get(3)), int(cap.get(4)))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(out_name +'.avi', fourcc, 30, frame_size)
                while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                                break
                        writer.write(detect(frame))
                cap.release()
else:
        # Show image with boxes
        output = detect(img)
        if len(sys.argv) == 4:
                cv2.imwrite(sys.argv[3] + ".jpg", output)
        cv2.imshow("Output", output)
        cv2.waitKey(0)

cv2.destroyAllWindows()