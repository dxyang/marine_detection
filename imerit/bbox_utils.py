"""
COCO: [x-px, y-px, w-px, h-px]
YOLO: [x-center, y-center, width-ratio, height-ratio]
"""

def yolo2coco_bbox():
    pass

def coco2yolo_bbox():
    pass

def labelbox2yolo_bbox(labelbox_dict, img_sz):
    img_w, img_h = img_sz
    x, y, w, h = labelbox_dict["left"], labelbox_dict["top"], labelbox_dict["width"], labelbox_dict["height"]

    yolo_x = (x + w/2) / img_w
    yolo_y = (y + h/2) / img_h
    yolo_w = w / img_w
    yolo_h = h / img_h

    return yolo_x, yolo_y, yolo_w, yolo_h
