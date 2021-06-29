import cv2


def get_class_names():
    # class_names = []
    class_file = 'ssd_mobilenet_v3_large_coco_2020_01_14/coco.names'
    with open(class_file, 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')
    print(class_names)
    return class_names


def init_model_network():
    config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weight_path = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weight_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    print('finished model set up')
    # exit()
    return net
