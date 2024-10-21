from os import path

import cv2

from ..src.connection_between_relationships import (
    DIRECTION,
    LINE_CLASSES,
    find_relationship,
    get_connection_between_relationships,
    get_first_movement,
    get_next_midpoint,
    get_next_movement_direction,
    is_relationship_connected,
    move_towards,
    search_connection,
)

TEST_FOLDER = path.dirname(path.abspath(__file__))
EXAMPLE_2_JPG_PATH = path.join(TEST_FOLDER, "img", "example_2.jpg")


def test_get_next_movement_direction():
    direction = get_next_movement_direction(LINE_CLASSES.CROSSING.value, DIRECTION.UP)
    assert direction == DIRECTION.UP

    direction = get_next_movement_direction(LINE_CLASSES.CROSSING.value, DIRECTION.DOWN)
    assert direction == DIRECTION.DOWN

    direction = get_next_movement_direction(LINE_CLASSES.CROSSING.value, DIRECTION.LEFT)
    assert direction == DIRECTION.LEFT

    direction = get_next_movement_direction(
        LINE_CLASSES.CROSSING.value, DIRECTION.RIGHT
    )
    assert direction == DIRECTION.RIGHT

    direction = get_next_movement_direction(
        LINE_CLASSES.HORIZONTAL.value, DIRECTION.RIGHT
    )
    assert direction == DIRECTION.RIGHT

    direction = get_next_movement_direction(
        LINE_CLASSES.HORIZONTAL.value, DIRECTION.LEFT
    )
    assert direction == DIRECTION.LEFT

    direction = get_next_movement_direction(
        LINE_CLASSES.BOTTOM_LEFT.value, DIRECTION.UP
    )
    assert direction == DIRECTION.LEFT

    direction = get_next_movement_direction(
        LINE_CLASSES.BOTTOM_LEFT.value, DIRECTION.RIGHT
    )
    assert direction == DIRECTION.DOWN

    direction = get_next_movement_direction(
        LINE_CLASSES.BOTTOM_LEFT.value, DIRECTION.LEFT
    )
    assert direction == DIRECTION.LEFT

    direction = get_next_movement_direction(
        LINE_CLASSES.TOP_LEFT.value, DIRECTION.RIGHT
    )
    assert direction == DIRECTION.UP

    direction = get_next_movement_direction(LINE_CLASSES.TOP_LEFT.value, DIRECTION.DOWN)
    assert direction == DIRECTION.LEFT

    direction = get_next_movement_direction(LINE_CLASSES.TOP_LEFT.value, DIRECTION.LEFT)
    assert direction == DIRECTION.LEFT

    direction = get_next_movement_direction(LINE_CLASSES.TOP_LEFT.value, DIRECTION.LEFT)
    assert direction == DIRECTION.LEFT

    direction = get_next_movement_direction(LINE_CLASSES.NONE.value, DIRECTION.LEFT)
    assert direction == None

    direction = get_next_movement_direction(
        LINE_CLASSES.BOTTOM_RIGHT.value, DIRECTION.UP
    )
    assert direction == DIRECTION.RIGHT

    direction = get_next_movement_direction(
        LINE_CLASSES.BOTTOM_RIGHT.value, DIRECTION.LEFT
    )
    assert direction == DIRECTION.DOWN

    direction = get_next_movement_direction(
        LINE_CLASSES.BOTTOM_RIGHT.value, DIRECTION.DOWN
    )
    assert direction == DIRECTION.DOWN

    direction = get_next_movement_direction(
        LINE_CLASSES.TOP_RIGHT.value, DIRECTION.DOWN
    )
    assert direction == DIRECTION.RIGHT

    direction = get_next_movement_direction(
        LINE_CLASSES.TOP_RIGHT.value, DIRECTION.LEFT
    )
    assert direction == DIRECTION.UP

    direction = get_next_movement_direction(
        LINE_CLASSES.TOP_RIGHT.value, DIRECTION.RIGHT
    )
    assert direction == DIRECTION.RIGHT

    direction = get_next_movement_direction(
        LINE_CLASSES.VERTICAL.value, DIRECTION.RIGHT
    )
    assert direction == DIRECTION.RIGHT


def test_find_relationship():
    entities = [
        {
            "index": 0,
            "relationships": [
                {"index": 0, "box": {"xmin": 10, "xmax": 20, "ymin": 10, "ymax": 20}},
                {"index": 1, "box": {"xmin": 30, "xmax": 40, "ymin": 30, "ymax": 40}},
            ],
        },
        {
            "index": 1,
            "relationships": [
                {"index": 0, "box": {"xmin": 50, "xmax": 60, "ymin": 50, "ymax": 60}},
                {"index": 1, "box": {"xmin": 70, "xmax": 80, "ymin": 70, "ymax": 80}},
            ],
        },
    ]

    midpoint = {"x": 15, "y": 15}
    entity_index, relationship_index = find_relationship(midpoint, entities)
    assert entity_index == 0
    assert relationship_index == 0

    midpoint = {"x": 35, "y": 35}
    entity_index, relationship_index = find_relationship(midpoint, entities)
    assert entity_index == 0
    assert relationship_index == 1

    midpoint = {"x": 55, "y": 55}
    entity_index, relationship_index = find_relationship(midpoint, entities)
    assert entity_index == 1
    assert relationship_index == 0

    midpoint = {"x": 75, "y": 75}
    entity_index, relationship_index = find_relationship(midpoint, entities)
    assert entity_index == 1
    assert relationship_index == 1

    midpoint = {"x": 15, "y": 999}
    entity_index, relationship_index = find_relationship(midpoint, entities)
    assert entity_index == None
    assert relationship_index == None

    midpoint = {"x": 999, "y": 15}
    entity_index, relationship_index = find_relationship(midpoint, entities)
    assert entity_index == None
    assert relationship_index == None

    midpoint = {"x": 999, "y": 999}
    entity_index, relationship_index = find_relationship(midpoint, entities)
    assert entity_index == None
    assert relationship_index == None


def test_move_towards():
    location = {"xmin": 100, "xmax": 200, "ymin": 100, "ymax": 200}
    displacement_measure = 50

    new_location = move_towards(
        current_location=location,
        displacement_measure=displacement_measure,
        direction=DIRECTION.UP,
        height=500,
        width=500,
    )
    assert new_location == {"xmin": 100, "xmax": 200, "ymin": 50, "ymax": 150}

    new_location = move_towards(
        current_location=location,
        displacement_measure=displacement_measure,
        direction=DIRECTION.DOWN,
        height=500,
        width=500,
    )
    assert new_location == {"xmin": 100, "xmax": 200, "ymin": 150, "ymax": 250}

    new_location = move_towards(
        current_location=location,
        displacement_measure=displacement_measure,
        direction=DIRECTION.LEFT,
        height=500,
        width=500,
    )
    assert new_location == {"xmin": 50, "xmax": 150, "ymin": 100, "ymax": 200}

    new_location = move_towards(
        current_location=location,
        displacement_measure=displacement_measure,
        direction=DIRECTION.RIGHT,
        height=500,
        width=500,
    )
    assert new_location == {"xmin": 150, "xmax": 250, "ymin": 100, "ymax": 200}


def test_get_next_midpoint():
    box = {"xmin": 100, "xmax": 200, "ymin": 100, "ymax": 200}

    midpoint = get_next_midpoint(box=box, direction=DIRECTION.UP)
    assert midpoint == {"x": 150, "y": 50}

    midpoint = get_next_midpoint(box=box, direction=DIRECTION.DOWN)
    assert midpoint == {"x": 150, "y": 250}

    midpoint = get_next_midpoint(box=box, direction=DIRECTION.LEFT)
    assert midpoint == {"x": 50, "y": 150}

    midpoint = get_next_midpoint(box=box, direction=DIRECTION.RIGHT)
    assert midpoint == {"x": 250, "y": 150}


def test_get_first_movement():
    entity_box = {"xmin": 100, "xmax": 200, "ymin": 100, "ymax": 200}

    relationship_box = {"xmin": 130, "xmax": 150, "ymin": 59, "ymax": 99}
    direction = get_first_movement(
        entity_box=entity_box, relationship_box=relationship_box
    )
    assert direction == DIRECTION.UP

    relationship_box = {"xmin": 130, "xmax": 150, "ymin": 201, "ymax": 241}
    direction = get_first_movement(
        entity_box=entity_box, relationship_box=relationship_box
    )
    assert direction == DIRECTION.DOWN

    relationship_box = {"xmin": 80, "xmax": 99, "ymin": 130, "ymax": 150}
    direction = get_first_movement(
        entity_box=entity_box, relationship_box=relationship_box
    )
    assert direction == DIRECTION.LEFT

    relationship_box = {"xmin": 201, "xmax": 221, "ymin": 130, "ymax": 150}
    direction = get_first_movement(
        entity_box=entity_box, relationship_box=relationship_box
    )
    assert direction == DIRECTION.RIGHT


def test_search_connection():
    img = cv2.imread(EXAMPLE_2_JPG_PATH)
    entities = [
        {
            "index": 0,
            "class": "ENTITY",
            "box": {"xmin": 123, "xmax": 235, "ymin": 1234, "ymax": 1344},
            "relationships": [
                {
                    "index": 0,
                    "class": "1M",
                    "box": {"xmin": 158, "xmax": 193, "ymin": 1199, "ymax": 1240},
                }
            ],
        },
        {
            "index": 1,
            "class": "ENTITY",
            "box": {"xmin": 1841, "xmax": 1949, "ymin": 83, "ymax": 190},
            "relationships": [
                {
                    "index": 0,
                    "class": "0M",
                    "box": {"xmin": 1877, "xmax": 1909, "ymin": 182, "ymax": 231},
                }
            ],
        },
    ]

    entity = {
        "index": 0,
        "class": "ENTITY",
        "box": {"xmin": 123, "xmax": 235, "ymin": 1234, "ymax": 1344},
        "relationships": [
            {
                "index": 0,
                "class": "1M",
                "box": {"xmin": 158, "xmax": 193, "ymin": 1199, "ymax": 1240},
            }
        ],
    }

    relationship = {
        "index": 0,
        "class": "1M",
        "box": {"xmin": 158, "xmax": 193, "ymin": 1199, "ymax": 1240},
    }

    connection = search_connection(
        entity=entity, relationship=relationship, entities=entities, img=img
    )
    assert connection == (1, 0)


def test_is_relationship_connected():
    connections = [
        {"entity_1": 0, "relationship_1": 0, "entity_2": 1, "relationship_2": 0},
        {"entity_1": 0, "relationship_1": 1, "entity_2": 1, "relationship_2": 1},
    ]

    is_connected = is_relationship_connected(
        connection_between_relationships=connections,
        entity_index=0,
        relationship_index=0,
    )
    assert is_connected

    is_connected = is_relationship_connected(
        connection_between_relationships=connections,
        entity_index=1,
        relationship_index=1,
    )
    assert is_connected

    is_connected = is_relationship_connected(
        connection_between_relationships=connections,
        entity_index=0,
        relationship_index=2,
    )
    assert not is_connected


def test_get_connection_between_relationships():
    img = cv2.imread(EXAMPLE_2_JPG_PATH)
    entities = [
        {
            "index": 0,
            "class": "ENTITY",
            "box": {"xmin": 123, "xmax": 235, "ymin": 1234, "ymax": 1344},
            "relationships": [
                {
                    "index": 0,
                    "class": "1M",
                    "box": {"xmin": 158, "xmax": 193, "ymin": 1199, "ymax": 1240},
                }
            ],
        },
        {
            "index": 1,
            "class": "ENTITY",
            "box": {"xmin": 1841, "xmax": 1949, "ymin": 83, "ymax": 190},
            "relationships": [
                {
                    "index": 0,
                    "class": "0M",
                    "box": {"xmin": 1877, "xmax": 1909, "ymin": 182, "ymax": 231},
                }
            ],
        },
    ]
    connections = get_connection_between_relationships(entities=entities, img=img)
    assert connections == [
        {"entity_1": 0, "relationship_1": 0, "entity_2": 1, "relationship_2": 0}
    ]
