from os import path

import cv2

from app.src.AI.ai_utils import is_prediction

from ..src.AI.entity.ai_entity import get_entities

TEST_FOLDER = path.dirname(path.abspath(__file__))
EXAMPLE_1_JPG_PATH = path.join(TEST_FOLDER, "img", "example_1.jpg")
EXAMPLE_2_JPG_PATH = path.join(TEST_FOLDER, "img", "example_2.jpg")


def test_example1():
    img = cv2.imread(EXAMPLE_1_JPG_PATH)
    entities = get_entities(img)
    entity_1 = {"class": "ENTITY", "midpoint": {"x": 100, "y": 270}}
    entity_2 = {"class": "ENTITY", "midpoint": {"x": 400, "y": 270}}
    entity_3 = {"class": "ENTITY", "midpoint": {"x": 400, "y": 50}}
    entity_4 = {"class": "ENTITY", "midpoint": {"x": 100, "y": 50}}
    not_entity = {"class": "ENTITY", "midpoint": {"x": 250, "y": 170}}

    assert len(entities) == 4
    assert any(
        is_prediction(entity_1, predicted_entity) for predicted_entity in entities
    )
    assert any(
        is_prediction(entity_2, predicted_entity) for predicted_entity in entities
    )
    assert any(
        is_prediction(entity_3, predicted_entity) for predicted_entity in entities
    )
    assert any(
        is_prediction(entity_4, predicted_entity) for predicted_entity in entities
    )
    assert not any(
        is_prediction(not_entity, predicted_entity) for predicted_entity in entities
    )


def test_example2():
    img = cv2.imread(EXAMPLE_2_JPG_PATH)
    entities = get_entities(img)
    entity_1 = {"class": "ENTITY", "midpoint": {"x": 1890, "y": 130}}
    entity_2 = {"class": "ENTITY", "midpoint": {"x": 170, "y": 1280}}
    not_entity = {"class": "ENTITY", "midpoint": {"x": 500, "y": 500}}

    assert len(entities) == 2
    assert any(
        is_prediction(entity_1, predicted_entity) for predicted_entity in entities
    )
    assert any(
        is_prediction(entity_2, predicted_entity) for predicted_entity in entities
    )
    assert not any(
        is_prediction(not_entity, predicted_entity) for predicted_entity in entities
    )
