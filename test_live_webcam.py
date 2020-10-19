from imageai.Detection.Custom import CustomVideoObjectDetection
from imageai.Detection.Custom import CustomObjectDetection
import os
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

# video_detector = CustomVideoObjectDetection()
video_detector = CustomObjectDetection()
video_detector.setModelTypeAsYOLOv3()
# video_detector.setModelPath("vending_machine_dual/models1/detection_model-ex-019--loss-0014.082.h5")
# video_detector.setJsonPath("vending_machine_dual/json/detection_config.json")
video_detector.setModelPath("E:/video_bebidas/models/detection_model-ex-048--loss-0001.200.h5")
video_detector.setJsonPath("E:/video_bebidas/json/detection_config.json")

video_detector.loadModel()

# video_detector.detectObjectsFromVideo(input_file_path="01瓶装百世可乐500毫升&康师傅冰糖雪梨500毫升右上.avi",
#                                           output_file_path=os.path.join(execution_path, "pepsi_kangshifu%"),
#                                           frames_per_second=30,
#                                           minimum_percentage_probability=50,
#                                           log_progress=True)

# video_detector.detectObjectsFromVideo(camera_input=camera,
#                                           output_file_path=os.path.join(execution_path, "cocaCola"),
#                                           frames_per_second=20,
#                                           minimum_percentage_probability=40,
#                                           log_progress=True)

while True:
    ## read frames
    ret, img = camera.read()
    ## predict yolo
    img, preds = video_detector.detectObjectsFromImage(input_image=img,
                      input_type="array",
                      output_type="array",
                      minimum_percentage_probability=60,
                      display_percentage_probability=True,
                      display_object_name=True)
    ## display predictions
    cv2.imshow("", img)
    ## press q or Esc to quit
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
## close camera
camera.release()
cv2.destroyAllWindows()