from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("E:/video_bebidas/models/detection_model-ex-058--loss-0001.168.h5") 
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "E:/video_bebidas\CocaCola_500ml\CocaCola_top2.mp4"),
                                output_file_path=os.path.join(execution_path, "Video")
                                , frames_per_second=30, log_progress=True)
print(video_path)