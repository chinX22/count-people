import torch
import cv2
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
                cap = cv2.VideoCapture(0)
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

def detect(frame, outframe):
        # Pass image through model, boxes collects coords for all directions, then
        # filter for only those with people
        output = model(frame)
        boxes = output.xyxy[0]
        peopleBoxes = boxes[boxes[:, 5] == 0]

        pop = len(peopleBoxes)

        # Draw boxes for every person found with colors based on confidence, blue = higher
        for *box, in peopleBoxes:
                cv2.rectangle(outframe, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                              (int(255 * box[4]), 0, int(255 * (1 - box[4]))), 2)
                
        cv2.putText(outframe, f"Found: {pop}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.3, (255, 255, 255), 2)
                
        return outframe



if sys.argv[1] == 'V':
        # If using a webcam, a frame rate limiter will be applied to reduce
        # latency in output
        if sys.argv[2] == "0":
                frameCount = 0
                while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                                break
                        #vv frame limiter for weaker machines
                        #if (frameCount % 2) == 0:
                        #        frameCount += 1
                        # Show image with boxes
                        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cv2.imshow("Output", detect(frameGray, frame))
                        if cv2.waitKey(1) == ord('q'):
                                break
                        #else : frameCount += 1
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
                        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        writer.write(detect(frameGray, frame))
                cap.release()
else:
        # Show image with boxes
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = detect(imgGray, img)
        if len(sys.argv) == 4:
                cv2.imwrite(sys.argv[3] + ".jpg", output)
        cv2.imshow("Output", output)
        cv2.waitKey(0)

cv2.destroyAllWindows()