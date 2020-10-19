import tensorflow as tf
from imageai.Detection.Custom import CustomObjectDetection

# execution_path = os.getcwd()
bbox=[]
scores=[]

x = tf.constant(bbox)
y = tf.constant(scores)

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
# detector.setModelPath("vending_machine/models3_tt2/detection_model-ex-010--loss-0006.916.h5")
# detector.setJsonPath("vending_machine/json3/detection_config.json")
detector.setModelPath("vending_machine/models_all20-11/detection_model-ex-010--loss-0008.401.h5")
detector.setJsonPath("vending_machine/json/detection_config.json")
detector.loadModel()

# all_images_array = []
#
# all_files = os.listdir(execution_path)
# for each_file in all_files:
#     if(each_file.endswith(".jpg") or each_file.endswith(".png")):
#         all_images_array.append(each_file)

# detections = detector.detectObjectsFromImage(input_image="549.jpg", output_image_path="549_detected.jpg",minimum_percentage_probability=40,nms_treshold=0.4)
detections = detector.detectObjectsFromImage(input_image="maidong1.jpg", output_image_path="maidong1_detected.jpg",minimum_percentage_probability=80)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

    bbox.append(detection["box_points"])
    scores.append(detection["percentage_probability"])
print("**********************************************")

detections = detector.detectObjectsFromImage(input_image="maidong2.jpg", output_image_path="maidong2_detected.jpg",
                                             minimum_percentage_probability=80)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

    bbox.append(detection["box_points"])
    scores.append(detection["percentage_probability"])
print("**********************************************")

detections = detector.detectObjectsFromImage(input_image="maidong3.jpg", output_image_path="maidong3_detected.jpg",
                                             minimum_percentage_probability=80)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

    bbox.append(detection["box_points"])
    scores.append(detection["percentage_probability"])
print("**********************************************")

detections = detector.detectObjectsFromImage(input_image="maidong4.jpg", output_image_path="maidong4_detected.jpg",
                                             minimum_percentage_probability=80)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

    bbox.append(detection["box_points"])
    scores.append(detection["percentage_probability"])
print("**********************************************")


# for detection in detections:
#     print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])



# print(int(detections['box_points']))
# print(bbox)
# print(scores)


# def non_max_suppression_with_tf(sess, boxes, scores, max_output_size, iou_threshold=0.5):
#     '''
#     Provide a tensorflow session and get non-maximum suppression
#
#     max_output_size, iou_threshold are passed to tf.image.non_max_suppression
#     '''
#     non_max_idxs = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=iou_threshold)
#     new_boxes = tf.cast(tf.gather(boxes, non_max_idxs), tf.int32)
#     new_scores = tf.gather(scores, non_max_idxs)
#     return sess.run([new_boxes, new_scores])
#
# # def nms_select(bbox,scores):
# #     selected_indices = tf.image.non_max_suppression(
# #        bbox, scores, 10, iou_threshold=0.5, score_threshold=0.1)
# #     selected_boxes = tf.gather(bbox, selected_indices)
# #     print(selected_boxes)
# #
# #     return selected_boxes
#
# with tf.Session() as sess:
# #    sess.run(nms_select(bbox, scores))
# #    # for detection in detections:
#     [selected_boxes, new_scores]=non_max_suppression_with_tf(sess,bbox,scores,10,0.5)
#     print(selected_boxes)
#     print(new_scores)
#
