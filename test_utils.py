from utility.preprocessing import parse_spyder_annotation, parse_mot_annotation
import matplotlib.pyplot as plt
import cv2
import random


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


opt  = input()

if opt == '1':
    imgs, seen_labels = parse_spyder_annotation('/home/arham/Documents/human-tracking/CV_training_data')
    plot_data(imgs[0])
    print(imgs[0])
elif opt == '2':
    imgs, seen_labels = parse_mot_annotation('/content/object-tracking/train', '')
    #plot_data(imgs[0])
    print(imgs[0])