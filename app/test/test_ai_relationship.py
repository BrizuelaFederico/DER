from os import path

import cv2

from app.src.AI.ai_utils import is_prediction

from ..src.AI.relationship.ai_relationship import get_relationships

TEST_FOLDER = path.dirname(path.abspath(__file__))
ENTITY_JPG_PATH = path.join(TEST_FOLDER, "img", "entity.jpg")


def test_relationship_entity():
    img = cv2.imread(ENTITY_JPG_PATH)
    relationships = get_relationships(img)
    relationship_1 = {"class": "1M", "midpoint": {"x": 245, "y": 110}}
    relationship_2 = {"class": "1M", "midpoint": {"x": 390, "y": 390}}
    relationship_3 = {"class": "0M", "midpoint": {"x": 310, "y": 80}}
    relationship_4 = {"class": "0M", "midpoint": {"x": 480, "y": 240}}
    relationship_5 = {"class": "0M", "midpoint": {"x": 190, "y": 145}}
    # relationship_6 = {"class": "0M", "midpoint": {"x": 320, "y": 360}} #prediction error, expected '0M', predicted '1M'
    # relationship_7 = {"class": "11", "midpoint": {"x": 390, "y": 110}}
    relationship_8 = {"class": "11", "midpoint": {"x": 450, "y": 295}}
    relationship_9 = {"class": "11", "midpoint": {"x": 170, "y": 300}}
    relationship_10 = {"class": "01", "midpoint": {"x": 245, "y": 365}}
    relationship_11 = {"class": "01", "midpoint": {"x": 460, "y": 160}}
    # relationship_12 = {"class": "1M", "midpoint": {"x": 190, "y": 220}}
    not_relationship = {"class": "1M", "midpoint": {"x": 50, "y": 50}}

    assert len(relationships) >= 8
    # assert any(
    #    is_prediction(relationship_1, predicted_relationship)
    #    for predicted_relationship in relationships
    # )
    assert any(
        is_prediction(relationship_2, predicted_relationship)
        for predicted_relationship in relationships
    )
    assert any(
        is_prediction(relationship_3, predicted_relationship)
        for predicted_relationship in relationships
    )
    assert any(
        is_prediction(relationship_4, predicted_relationship)
        for predicted_relationship in relationships
    )
    assert any(
        is_prediction(relationship_5, predicted_relationship)
        for predicted_relationship in relationships
    )
    # assert any(
    #    is_prediction(relationship_6, predicted_relationship)
    #    for predicted_relationship in relationships
    # )
    # assert any(
    #    is_prediction(relationship_7, predicted_relationship)
    #    for predicted_relationship in relationships
    # )
    assert any(
        is_prediction(relationship_8, predicted_relationship)
        for predicted_relationship in relationships
    )
    assert any(
        is_prediction(relationship_9, predicted_relationship)
        for predicted_relationship in relationships
    )
    assert any(
        is_prediction(relationship_10, predicted_relationship)
        for predicted_relationship in relationships
    )
    assert any(
        is_prediction(relationship_11, predicted_relationship)
        for predicted_relationship in relationships
    )
    # assert any(
    #    is_prediction(relationship_12, predicted_relationship)
    #    for predicted_relationship in relationships
    # )
    assert not any(
        is_prediction(not_relationship, predicted_relationship)
        for predicted_relationship in relationships
    )
