import numpy as np


# for entity and relationship
def get_detections(img, sess, outputs, min_score, classes):
    img_array = np.array(img)
    img_array = np.expand_dims(img_array.astype(np.uint8), axis=0)
    num_detections, detection_boxes, detection_scores, detection_classes = sess.run(
        outputs, {"input_tensor": img_array}
    )
    detections = filter_detections_by_score(
        detection_boxes[0], detection_scores[0], detection_classes[0], min_score
    )

    return convert_detections_to_listdict(detections, img_array, classes)


def filter_detections_by_score(
    detection_boxes, detection_scores, detection_classes, min_score
):
    detections = list(zip(detection_boxes, detection_scores, detection_classes))
    return filter(lambda detection: detection[1] > min_score, detections)


def convert_detections_to_listdict(detections, img_array, classes):
    listdict = []
    for index, detection in enumerate(detections):
        listdict.append(convert_detection_to_dict(index, detection, img_array, classes))
    return listdict


def convert_detection_to_dict(index, detection, img_array, classes):
    return {
        "index": index,
        "class": classes[int(detection[2] - 1)],
        "box": get_region(detection[0], img_array),
    }


def get_region(box, img_array):
    height = img_array.shape[1]
    width = img_array.shape[2]
    return {
        "xmin": int(box[1] * width),
        "xmax": int(box[3] * width),
        "ymin": int(box[0] * height),
        "ymax": int(box[2] * height),
    }


def is_prediction(expected, predicted):
    return expected["class"] == predicted["class"] and is_midpoint_inside_box(
        expected["midpoint"], predicted["box"]
    )


def is_midpoint_inside_box(midpoint, box):
    return (
        box["xmin"] < midpoint["x"] < box["xmax"]
        and box["ymin"] < midpoint["y"] < box["ymax"]
    )


def expand_box(box, width, height):
    x_axis_displacement = int((box["xmax"] - box["xmin"]) / 2)
    y_axis_displacement = int((box["ymax"] - box["ymin"]) / 2)
    return {
        "xmin": max(0, box["xmin"] - x_axis_displacement),
        "xmax": min(width, box["xmax"] + x_axis_displacement),
        "ymin": max(0, box["ymin"] - y_axis_displacement),
        "ymax": min(height, box["ymax"] + y_axis_displacement),
    }


def crop_image(box, img):
    return img[box["ymin"] : box["ymax"], box["xmin"] : box["xmax"]]
