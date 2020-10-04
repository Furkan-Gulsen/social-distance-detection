import cv2
import time
import numpy as np


weightsPath = "./yolov3.weights"
configPath = "./yolov3.cfg"
coco_names = "./coco.names"
LABELS = open(coco_names).read().strip().split("\n")
video_path = "./videos/video.mp4"



def load_input_image(image_path):
    test_img = cv2.imread(image_path)
    h, w, _ = test_img.shape
    return test_img, h, w



def yolov3(yolo_weights, yolo_cfg, coco_names):
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    classes = open(coco_names).read().strip().split("\n")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers



def get_calibrate(p1,p2):
  dist0 = p1[0] - p2[0]
  sum1 = p1[1] + p2[1]
  dist1 = p1[1] - p2[1]
  calibrate = (dist0 ** 2 + (550 / (sum1 / 2)) * (dist1 ** 2)) ** 0.5
  return calibrate


"""
  0 => Safe
  1 => Low Risk
  2 => High Risk
"""
def distance(p1,p2) -> int:
  cd = get_calibrate(p1,p2)
  calibrate = (p1[1] + p2[1])/2
  if 0 < cd < 0.15 * calibrate:
    return 2
  elif 0 < cd < 0.20 * calibrate:
    return 1
  else:
    return 0


progress = 0
count = 0


def perform_detection(net, img, output_layers, w, h, confidence_threshold):
    global progress, count
    blob = cv2.dnn.blobFromImage(img, 1 / 255., (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    person_cords = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if LABELS[class_id] == 'person':
              if confidence > confidence_threshold:
                  center_x, center_y, width, height = list(map(int, 
                  	detection[0:4] * [w, h, w, h]))
                  # print("centerX: ", center_x)
                  person_cords.append((center_x,center_y))
                
                  top_left_x = int(center_x - (width / 2))
                  top_left_y = int(center_y - (height / 2))

                  boxes.append([top_left_x, top_left_y, width, height])
                  confidences.append(float(confidence))
                  class_ids.append(class_id)
    
    if progress == 24:
      count += 1
      print("Video Time: {}".format(count))
      progress = 0
    else:
      progress += 1

    
    # print("person_cords: ", person_cords)
    return boxes, confidences, class_ids




def draw_boxes(boxes, confidences, class_ids, classes, img, 
	colors, confidence_threshold, NMS_threshold):

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (169,121,25), 2)
            # cv2.circle(img, (int(x+(w/2)), int(y+(h/2))), 2, (0,255,0), 2 )
            # text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            # cv2.putText(img, text, (x, y - 5), FONT, 0.5, color, 2)

    return indexes




def detection_image_file(image_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):

    img, h, w = load_input_image(image_path)
    net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    boxes, confidences, class_ids = perform_detection(net, img, output_layers, w, h, confidence_threshold)
    draw_boxes(boxes, confidences, class_ids, classes, img, colors, confidence_threshold, nms_threshold)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




def calculate_dist(boxes, indexes):

  if len(indexes) > 0:
    idf = indexes.flatten()
    centers = list()
    status = list()

    safe = list()
    low_risk = list()
    high_risk = list()

    for i in idf:
      (x,y) = (boxes[i][0],boxes[i][1]) # top-left position
      (w,h) = (boxes[i][2],boxes[i][3])
      centers.append([int(x+(w/2)), int(y+(h/2))])
      status.append(0)
    
    cr = range(len(centers))
    for c1 in cr:
      for c2 in cr:
        dst = distance(centers[c1], centers[c2])
        if dst == 2:
          high_risk.append([centers[c1], centers[c2]])
          status[c1] = 2
          status[c2] = 2
        elif dst == 1:
          low_risk.append([centers[c1], centers[c2]])
          if status[c1] != 2:
            status[c1] = 1
          if status[c2] != 2:
            status[c2] = 1

    person_count = len(centers)
    high_risk_count = status.count(2)
    low_risk_count = status.count(1)
    safe_count = status.count(0)
    return high_risk_count, low_risk_count, safe_count, idf, high_risk, low_risk





def draw_distance(frame, idf, boxes, WIDTH, HEIGHT, high_risk, low_risk, 
	safe, high_risk_cord, low_risk_cord):

  for i in idf:
    sub_img = frame[600:HEIGHT, 0:200]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.70, black_rect, 0.30, 1.0)
    frame[600:HEIGHT, 0:200] = res
    """
    sub_img = frame[10:520, 230:710]
    frame[10:520, 230:710] = res
    """
    cv2.putText(frame, "TOTAL    : {}".format(high_risk+low_risk+safe),(25, HEIGHT - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "HIGH RISK: {}".format(high_risk),(25, HEIGHT - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
    cv2.putText(frame, "LOW RISK : {}".format(low_risk), (25, HEIGHT - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
    cv2.putText(frame, "SAFE     : {}".format(safe), (25, HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255, 255, 255), 2)
    
  for l in high_risk_cord:
    cv2.line(frame, tuple(l[0]), tuple(l[1]), (10, 44, 236), 2)
  for h in low_risk_cord:
    cv2.line(frame, tuple(h[0]), tuple(h[1]), (10, 236, 236), 2)
  
  return frame




def detection_video_file(video_path, yolo_weights, yolo_cfg, coco_names, 
	confidence_threshold, nms_threshold):

  net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  video = cv2.VideoCapture(video_path)
  writer = None
  (HEIGHT,WIDTH) = (None, None)
  MAX_WIDTH = 0

  while True:
    (ret, frame) = video.read()

    if not ret:
      break
      
    if WIDTH is None or HEIGHT is None:
      (HEIGHT,WIDTH) = frame.shape[:2]
      MAX_WIDTH = WIDTH
    
    frame = frame[0:HEIGHT, 200:MAX_WIDTH]
    (HEIGHT,WIDTH) = frame.shape[:2]
    
    h, w = frame.shape[:2]
    boxes, confidences, class_ids = perform_detection(net, frame, output_layers, w, h, confidence_threshold)
    indexes = draw_boxes(boxes, confidences, class_ids, classes, frame, colors, confidence_threshold, nms_threshold)
    high_risk, low_risk, safe, idf, high_risk_cord, low_risk_cord = calculate_dist(boxes, indexes)
    draw_distance(frame, idf, boxes, WIDTH, HEIGHT, high_risk, low_risk, safe, high_risk_cord, low_risk_cord)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter("output.avi", fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    writer.write(frame)

  writer.release()
  video.release()



# run code
detection_video_file(video_path, weightsPath, configPath, coco_names, 0.5, 0.5)
