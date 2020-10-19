from imageai.Detection import VideoObjectDetection
from imageai.Detection import ObjectDetection

from imageai.Detection.Custom import CustomVideoObjectDetection

import os
import cv2

execution_path = os.getcwd()

# camera = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

# detector = VideoObjectDetection()
# detector = ObjectDetection()
detector = CustomVideoObjectDetection()
detector.setModelTypeAsYOLOv3()
# detector.setModelPath(os.path.join(execution_path , "yolo.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
detector.setModelPath("E:/video_bebidas/models/detection_model-ex-048--loss-0001.200.h5")
detector.setJsonPath("E:/video_bebidas/json/detection_config.json")
detector.loadModel()
#
# video_path = detector.detectObjectsFromVideo(camera_input=camera,
#                                 output_file_path=os.path.join(execution_path, "camera_detected_video")
#                                 , frames_per_second=20, log_progress=True, minimum_percentage_probability=30)
# print(video_path)


while True:
    ## read frames
    ret, img = cam.read()
    ## predict yolo
    img, preds = detector.detectCustomObjectsFromImage(input_image=img,
                      custom_objects=None, input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=False,
                      display_object_name=True)
    ## display predictions
    cv2.imshow("", img)
    ## press q or Esc to quit
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
## close camera
cam.release()
cv2.destroyAllWindows()