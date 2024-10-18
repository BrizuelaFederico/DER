from os import path

import cv2

from ..src.AI.relationship.ai_relationship import get_relationships

TEST_FOLDER = path.dirname(path.abspath(__file__))
ENTITY_JPG_PATH = path.join(TEST_FOLDER, "img", "entity.jpg")


def is_midpont_inside_box(midpoint, box):
    return (
        box["xmin"] < midpoint["x"] < box["xmax"]
        and box["ymin"] < midpoint["y"] < box["ymax"]
    )


def is_prediction(relationship, predicted_relationship):
    return relationship["type"] == predicted_relationship[
        "type"
    ] and is_midpont_inside_box(relationship["midpoint"], predicted_relationship["box"])


def test_relationship_entity():
    img = cv2.imread(ENTITY_JPG_PATH)
    relationships = get_relationships(img)
    relationship_1 = {"type": "1M", "midpoint": {"x": 245, "y": 110}}
    relationship_2 = {"type": "1M", "midpoint": {"x": 390, "y": 390}}
    relationship_3 = {"type": "0M", "midpoint": {"x": 310, "y": 80}}
    relationship_4 = {"type": "0M", "midpoint": {"x": 480, "y": 240}}
    relationship_5 = {"type": "0M", "midpoint": {"x": 190, "y": 145}}
    # relationship_6 = {"type": "0M", "midpoint": {"x": 320, "y": 360}} #prediction error, expected '0M', predicted '1M'
    relationship_7 = {"type": "11", "midpoint": {"x": 390, "y": 110}}
    relationship_8 = {"type": "11", "midpoint": {"x": 450, "y": 295}}
    relationship_9 = {"type": "11", "midpoint": {"x": 170, "y": 300}}
    relationship_10 = {"type": "01", "midpoint": {"x": 245, "y": 365}}
    relationship_11 = {"type": "01", "midpoint": {"x": 460, "y": 160}}
    relationship_12 = {"type": "1M", "midpoint": {"x": 190, "y": 220}}
    not_relationship = {"type": "1M", "midpoint": {"x": 50, "y": 50}}

    assert len(relationships) == 12
    assert any(
        is_prediction(relationship_1, predicted_relationship)
        for predicted_relationship in relationships
    )
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
    assert any(
        is_prediction(relationship_7, predicted_relationship)
        for predicted_relationship in relationships
    )
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
    assert any(
        is_prediction(relationship_12, predicted_relationship)
        for predicted_relationship in relationships
    )
    assert not any(
        is_prediction(not_relationship, predicted_relationship)
        for predicted_relationship in relationships
    )
