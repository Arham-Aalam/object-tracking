from utility.preprocessing import parse_spyder_annotation, parse_mot_annotation
import matplotlib.pyplot as plt
import cv2
import random
from utility.preprocessing import parse_annotation, parse_mot_annotation, BatchSequenceGenerator1
from utility.utils import WeightReader, decode_netout, draw_boxes, normalize
import numpy as np

def plot_data(obj):
    img = cv2.imread(obj['filename'])
    for o in obj['object']:
        x1, y1, x2, y2 = list(o.values())[1:]
        r, g, b = [random.randint(0, 255) for _ in range(3)]
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (r, g, b), 3)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imwrite('img.jpg', img)
    plt.imsave('img.jpg', img)


LABELS_MOT17 =  [
                        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
                    ]

LABELS           = LABELS_MOT17
IMAGE_H, IMAGE_W = 416, 416 # 416 Dimention issue
GRID_H,  GRID_W  = 13 , 13  # 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.5 #0.3
NMS_THRESHOLD    = 0.45 #0.3
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 1
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50

SEQUENCE_LENGTH   = 4
MAX_BOX_PER_IMAGE = 50

#LOAD_MODEL        = True
LOAD_MODEL        = False
INITIAL_EPOCH     = 0
SAVED_MODEL_PATH  = 'models/MultiObjDetTracker-CHKPNT-03-0.55.hdf5'

# train_image_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Data/VID/train/'
# train_annot_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Annotations/VID/train/'
# valid_image_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Data/VID/val/'
# valid_annot_folder = 'data/ImageNet-ObjectDetection/ILSVRC2015Train/Annotations/VID/val/'

# train_image_folder = 'data/MOT17/MOT17Det/train/'
# train_annot_folder = 'data/MOT17Ann/train/'
# valid_image_folder = 'data/MOT17/MOT17Det/train/'
# valid_annot_folder = 'data/MOT17Ann/val/'

train_image_folder = 'train/'
train_annot_folder = 'train/'
valid_image_folder = 'test/'
valid_annot_folder = 'test/'

model          = None
detector       = None
model_detector = None

opt  = input()

if opt == '1':
    imgs, seen_labels = parse_spyder_annotation('/home/arham/Documents/human-tracking/CV_training_data')
    plot_data(imgs[0])
    print(imgs[0])
elif opt == '2':
    imgs, seen_labels = parse_mot_annotation('/content/object-tracking/train', '', 't')
    #plot_data(imgs[0])
    print(len(imgs))
    print(imgs[0], imgs[1])
    print(seen_labels)
    imgs2, seen_labels2 = parse_mot_annotation('/content/object-tracking/train', '', 'v')
    #plot_data(imgs[0])
    print(len(imgs2))
    print(imgs2[0], imgs2[1])
    print(seen_labels2)

    generator_config = {
            'IMAGE_H'         : IMAGE_H,
            'IMAGE_W'         : IMAGE_W,
            'GRID_H'          : GRID_H,
            'GRID_W'          : GRID_W,
            'BOX'             : BOX,
            'LABELS'          : LABELS,
            'CLASS'           : len(LABELS),
            'ANCHORS'         : ANCHORS,
            'BATCH_SIZE'      : BATCH_SIZE,
            'TRUE_BOX_BUFFER' : 50,
            'SEQUENCE_LENGTH' : SEQUENCE_LENGTH
        }

    train_batch = BatchSequenceGenerator1(imgs, generator_config, norm=normalize, shuffle=True, augment=True)
    valid_batch = BatchSequenceGenerator1(imgs2, generator_config, norm=normalize, augment=False)

    print(len(train_batch))
    print(len(valid_batch))

    print(train_batch.__getitem__(0))
    print(valid_batch.__getitem__(0))