from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="E:/video_bebidas")
# trainer.setTrainConfig(object_names_array=["CocaCola500ml"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.setTrainConfig(object_names_array=["CocaCola500ml"], batch_size=4, num_experiments=100, train_from_pretrained_model="E:/video_bebidas/models/detection_model-ex-058--loss-0001.168.h5")
trainer.trainModel()

# tensorboard --logdir video_bebidas\logs
