from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="vending_machine_dual")
metrics = trainer.evaluateModel(model_path="vending_machine_dual/models", json_path="vending_machine_dual/json/detection_config.json", iou_threshold=0.5, object_threshold=0.5, nms_threshold=0.5)