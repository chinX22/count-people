import torch
import cv2
import random

# Load pre trained yolov5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# load image
source = "class3.jpg"
img = cv2.imread(source)

# Pass image through model, boxes collects coords for all directions, then
# filter for only those with people
output = model(img)
boxes = output.xyxy[0]
peopleBoxes = boxes[boxes[:, 5] == 0]

# Get and print the number of people found
pop = len(peopleBoxes)
print(f"people found: {pop}")

# Draw boxes for every person found with varying colors
for *box, in peopleBoxes:
        rand = random.randint(0, 255)
        box = list(map(int, box))
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (rand, 255 - rand, 0), 2)

# Show image with boxes
cv2.imshow("Output", img)
cv2.waitKey(0)