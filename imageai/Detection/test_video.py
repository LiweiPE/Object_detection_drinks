from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("vending_machine_dual/models1/detection_model-ex-019--loss-0014.082.h5")
video_detector.setJsonPath("vending_machine_dual/json/detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="01瓶装百世可乐500毫升&康师傅冰糖雪梨500毫升右上.avi",
                                          output_file_path=os.path.join(execution_path, "pepsi_kangshifu%"),
                                          frames_per_second=30,
                                          minimum_percentage_probability=50,
                                          log_progress=True)