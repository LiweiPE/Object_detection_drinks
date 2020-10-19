from imageai.Detection.Custom import DetectionModelTrainer

# pretrained-yolov3.h5
trainer = DetectionModelTrainer()
# trainer.setModelTypeAsResNet()
trainer.setModelTypeAsYOLOv3()
# trainer.setGpuUsage(train_gpus="0,1,2,3,4,5,6,7,8")
trainer.setGpuUsage(train_gpus="0,1,2,3,4,5,6,7")
trainer.setDataDirectory(data_directory="vending_machine")
# trainer.setTrainConfig(object_names_array=["orange_juice"], batch_size=8, num_experiments=10,train_from_pretrained_model="pretrained-yolov3.h5")
trainer.setTrainConfig(object_names_array=["pepsi","water","orange_juice","cucumber_soda","C100_juice","pepsi_330","HongNiu","Wangzi_milk","Wanglaoji","Beibingyang","Asamu_milktea","Harbin_beer","Kangshifu_juice","Maidong_lime","Dongfang_greentea"],
                       batch_size=32, num_experiments=10,train_from_pretrained_model="vending_machine/models_all_25-11/detection_model-ex-010--loss-0007.615.h5")
# trainer.setTrainConfig(object_names_array=["pepsi_coke"], batch_size=1, num_experiments=50,show_network_summary=True)
# In the above,when training for detecting multiple objects,
#set object_names_array=["object1", "object2", "object3",..."objectz"]
trainer.trainModel()
#
#