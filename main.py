import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import dlib
import logging
import time
import threading


# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(level = logging.INFO, format = "[INFO] %(message)s")
logger = logging.getLogger(__name__)

model = YOLO('yolov8x.pt')


## Input Video
test_video = 'Input\input.mp4'
logger.info("Starting the video..")
cap = cv2.VideoCapture(test_video)

##for camera ip
# camera_ip = "Camera Url"
# logger.info("Starting the live stream..")
# cap = cv2.VideoCapture(camera_ip)
# time.sleep(1.0)



my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


#function for detect person coordinate
def get_person_coordinates(frame):
    """
    Extracts the coordinates of the person bounding boxes from the YOLO model predictions.

    Args:
        frame: Input frame for object detection.

    Returns:
        list: List of person bounding box coordinates in the format [x1, y1, x2, y2].
    """
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data.detach().cpu()
    px = pd.DataFrame(a).astype("float")

    list_corr = []
    for index, row in px.iterrows():
        x1 = row[0]
        y1 = row[1]
        x2 = row[2]
        y2 = row[3]
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list_corr.append([x1, y1, x2, y2])
    return list_corr


def people_counter():
    """
    Counts the number of people entering and exiting based on object tracking.
    """
    count = 0

    writer = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=40)
    trackers = []
    trackableObjects = {}

    # Initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # Initialize empty lists to store the counting data
    total = []
    move_out = []
    move_in = []

    # Initialize video writer
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('Final_output.mp4', fourcc, 30, (W, H), True)

    fps = FPS().start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (500, 280))

        per_corr = get_person_coordinates(frame)

        rects = []
        if totalFrames % 30 == 0:
            trackers = []
            for bbox in per_corr:
                x1, y1, x2, y2 = bbox
                rects.append([x1, y1, x2, y2])
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                tracker.start_track(frame, rect)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
                trackers.append(tracker)
        else:
            for tracker in trackers:
                tracker.update(frame)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, H // 2 - 10), (W, H // 2 - 10), (0, 0, 0), 2)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2 - 20:
                        totalUp += 1
                        move_out.append(totalUp)
                        to.counted = True
                    elif 0 < direction < 1.1 and centroid[1] > 144:
                        totalDown += 1
                        move_in.append(totalDown)
                        to.counted = True

                        total = []
                        total.append(len(move_in) - len(move_out))

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info_status = [
            ("Enter", totalUp),
            ("Exit ", totalDown),
        ]

        # info_total = [("Total people inside", ', '.join(map(str, total)))]
        
        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        writer.write(frame)
        cv2.imshow("People Count", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        totalFrames += 1
        fps.update()

        end_time = time.time()
        num_seconds = (end_time - start_time)
        if num_seconds > 28800:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))



if __name__ == "__main__":
    people_counter()
    

## Apply threading also

# def start_people_counter():
#     t1 = threading.Thread(target=people_counter)
#     t1.start()


# if __name__ == "__main__":
#     start_people_counter()




