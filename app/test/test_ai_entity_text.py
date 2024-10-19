from os import path

import cv2

from app.src.AI.text.ai_entity_text import get_entity_text

TEST_FOLDER = path.dirname(path.abspath(__file__))
ENTITY_JPG_PATH = path.join(TEST_FOLDER, "img", "entity.jpg")
ENTITY_2_JPG_PATH = path.join(TEST_FOLDER, "img", "entity_2.jpg")


def test_entity_text():
    img = cv2.imread(ENTITY_JPG_PATH)
    entity_text = get_entity_text(img)
    assert len(entity_text["name"]) > 5
    assert len(entity_text["attributes"]) == 1


def test_entity_2_text():
    img = cv2.imread(ENTITY_2_JPG_PATH)
    entity_text = get_entity_text(img)
    assert len(entity_text["name"]) > 5
    assert len(entity_text["attributes"]) == 4
