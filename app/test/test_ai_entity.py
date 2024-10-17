from os import path

import cv2

from ..src.AI.entity.ai_entity import get_entity_boxes

TEST_FOLDER = path.dirname(path.abspath(__file__))
EXAMPLE_1_JPG_PATH = path.join(TEST_FOLDER, "img", "example_1.jpg")
EXAMPLE_2_JPG_PATH = path.join(TEST_FOLDER, "img", "example_2.jpg")


def is_midpont_inside_box(midpoint, box):
    midpoint_y = midpoint[0]
    midpoint_x = midpoint[1]
    box_xmin = box[1]
    box_xmax = box[3]
    box_ymin = box[0]
    box_ymax = box[2]
    return box_xmin < midpoint_x < box_xmax and box_ymin < midpoint_y < box_ymax


def test_example1():
    img = cv2.imread(EXAMPLE_1_JPG_PATH)
    boxes = get_entity_boxes(img)
    midpoint_1 = [270, 100]
    midpoint_2 = [270, 400]
    midpoint_3 = [50, 400]
    midpoint_4 = [50, 100]
    not_midpoint = [170, 250]
    assert len(boxes) == 4
    assert any(is_midpont_inside_box(midpoint_1, box) for box in boxes)
    assert any(is_midpont_inside_box(midpoint_2, box) for box in boxes)
    assert any(is_midpont_inside_box(midpoint_3, box) for box in boxes)
    assert any(is_midpont_inside_box(midpoint_4, box) for box in boxes)
    assert not any(is_midpont_inside_box(not_midpoint, box) for box in boxes)


def test_example2():
    img = cv2.imread(EXAMPLE_2_JPG_PATH)
    boxes = get_entity_boxes(img)
    midpoint_1 = [130, 1890]
    midpoint_2 = [1280, 170]
    not_midpoint = [500, 500]
    assert len(boxes) == 2
    assert any(is_midpont_inside_box(midpoint_1, box) for box in boxes)
    assert any(is_midpont_inside_box(midpoint_2, box) for box in boxes)
    assert not any(is_midpont_inside_box(not_midpoint, box) for box in boxes)
