from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
# trainer.setModelTypeAsResNet()
trainer.setModelTypeAsYOLOv3()
trainer.setGpuUsage(train_gpus="0,1,2,3,4,5,6,7")
trainer.setDataDirectory(data_directory="vending_machine_dual")
# "北冰洋细灌装橙汁汽水330毫升",
trainer.setTrainConfig(object_names_array=["北冰洋细罐装橙汁汽水330毫升","康师傅冰糖雪梨500毫升","百事可乐瓶装500毫升","百岁山饮用天然矿泉水570毫升","东方树叶绿茶500毫升","美汁源果粒橙420毫升","水溶C100西柚汁饮料445毫升","脉动水蜜桃口味600毫升","红牛维生素功能饮料250毫升","旺仔牛奶罐装245毫升","百事可乐罐装330毫升","王老吉罐装310毫升","茶π蜜桃乌龙茶500毫升","茶π蜜桃乌龙茶900毫升","统一阿萨姆原味奶茶500毫升","哈尔滨啤酒小麦王500毫升","元气森林青瓜味苏打气泡水480毫升"],
                       batch_size=32, num_experiments=50,train_from_pretrained_model="pretrained-yolov3.h5")
# trainer.setTrainConfig(object_names_array=["pepsi_coke"], batch_size=1, num_experiments=50,show_network_summary=True)
# In the above,when training for detecting multiple objects,
#set object_names_array=["object1", "object2", "object3",..."objectz"]
trainer.trainModel()
#
#