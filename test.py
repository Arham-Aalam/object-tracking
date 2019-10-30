import os
import json
import importlib
from utility.utils import prepare_data
from models_detection.KerasYOLO import KerasYOLO
from models_tracking.MultiObjDetTracker import MultiObjDetTracker

def single_object_tracking():
    with open("config.json") as config_buffer:
        config = json.loads(config_buffer.read())

    tracker_name  = config["model_tracker"]["name"]
    tracker_class = getattr(importlib.import_module("models_tracking." + tracker_name), tracker_name)
    tracker       = tracker_class()

    tracker.train()

def simult_multi_obj_detection_tracking():
    model = MultiObjDetTracker()
    model.train()

def keras_yolo_obj_detection():
    model = KerasYOLO()
    model.train()

    prefix = 'darknet/data/'
    inputs = ['dog.jpg', 'eagle.jpg', 'giraffe.jpg', 'horses.jpg', 'person.jpg']
    model = KerasYOLO()
    for input_instance in inputs:
        model.predict(prefix + input_instance, input_instance)

def test_multitracker(input_path, output_path):
    files_arr, out_arr = [], []
    cnt = 1
    for file in os.listdir(input_path):
        file_path = os.path.join(input_path, file)
        out_path = os.path.join(output_path, 'out_{}.jpg'.format(cnt))
        files_arr.append(file_path)
        out_arr.append(out_path)
        cnt += 1
    model = MultiObjDetTracker()
    model.load_weights()
    model.predict(files_arr, out_arr)

if __name__=='__main__':

    if not os.path.exists('logs'):
        os.mkdir('logs/')
    if not os.path.exists('models'):
        os.mkdir('models/')
    
    test_multitracker('/object-tracking/test_imgs', '/object-tracking/output_imgs')