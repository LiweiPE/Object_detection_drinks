from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("E:/video_bebidas/models/detection_model-ex-028--loss-0001.296.h5")
video_detector.setJsonPath("E:/video_bebidas\json/detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="E:/video_bebidas\CocaCola_500ml\CocaCola_top1.mp4",
                                          output_file_path=os.path.join(execution_path, "Video_coca50%_ex28"),
                                          frames_per_second=30,
                                          minimum_percentage_probability=50,
                                          log_progress=True)