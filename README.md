# LiweiPE-Object_detection_drinks
This reasearch was develop to detect many soft drinks or snacks taking out from vending machine.

### Configuration environment

conda activate tf1.15
* pip install imageio
### Dataset
Images annotations have to be changed to xmal format
Whole raw dataset can be found at /data/vending
Training dataset to detect one object can be found at /data/home/liwei/ImageAI/vending_machine
Training dataset to detect more than two objects at the same time can be found at /data/home/liwei/ImageAI/vending_machine_dual

### Training
The following example is used to detect different custom objects

```sh
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setGpuUsage(train_gpus="0,1,2,3,4,5,6,7")
trainer.setDataDirectory(data_directory="vending_machine")
# trainer.setTrainConfig(object_names_array=["orange_juice"], batch_size=8, num_experiments=10,train_from_pretrained_model="pretrained-yolov3.h5")
trainer.setTrainConfig(object_names_array=["pepsi","water","orange_juice","cucumber_soda","C100_juice","pepsi_330","HongNiu","Wangzi_milk","Wanglaoji","Beibingyang","Asamu_milktea","Harbin_beer","Kangshifu_juice","Maidong_lime","Dongfang_greentea"],
                       batch_size=32, num_experiments=10,train_from_pretrained_model="vending_machine/models_all_25-11/detection_model-ex-010--loss-0007.615.h5")
trainer.trainModel()
```

Training file for one custom object detection

```sh
$ python vending_machine.py
```
Training file for multiple custom object detection

```sh
$ python vending_machine_dual.py
```
### Test
```sh
$ python test_video.py
```
Example for testing video:
```sh
from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()
video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("vending_machine_dual/models1/detection_model-ex-039--loss-0011.630.h5")
video_detector.setJsonPath("vending_machine_dual/json1/detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="01鐡惰鐧句笘鍙箰500姣崌&搴峰笀鍌呭啺绯栭洩姊?00姣崌鍙充笂.avi",
                                          output_file_path=os.path.join(execution_path, "pepsi_kangshifu%"),
                                          frames_per_second=30,
                                          minimum_percentage_probability=50,
                                          log_progress=True)
```

### Citation
@misc {ImageAI,
    author = "Moses and John Olafenwa",
    title  = "ImageAI, an open source python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities",
    url    = "https://github.com/OlafenwaMoses/ImageAI",
    month  = "mar",
    year   = "2018--"
}


 
