from enum import Enum

from .AI.ai_utils import is_midpoint_inside_box
from .AI.line.ai_line import recognize_line


class LINE_CLASSES(Enum):
    CROSSING = "CROSSING"
    HORIZONTAL = "HORIZONTAL"
    BOTTOM_LEFT = "BOTTOM-LEFT"
    TOP_LEFT = "TOP-LEFT"
    NONE = "NONE"
    BOTTOM_RIGHT = "BOTTOM-RIGHT"
    TOP_RIGHT = "TOP-RIGHT"
    VERTICAL = "VERTICAL"


class DIRECTION(Enum):
    UP = "UP"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    LEFT = "LEFT"


def get_connection_between_relationships(entities, img):
    connection_between_relationships = []
    for entity in entities:
        for relationship in entity["relationships"]:
            if is_relationship_connected(
                entity_index=entity["index"],
                relationship_index=relationship["index"],
                connection_between_relationships=connection_between_relationships,
            ):
                continue

            entity_index, relationship_index = search_connection(
                entity=entity,
                relationship=relationship,
                entities=entities,
                img=img,
            )

            if not entity_index:
                continue

            connection_between_relationships.append(
                {
                    "entity_1": entity["index"],
                    "relationship_1": relationship["index"],
                    "entity_2": entity_index,
                    "relationship_2": relationship_index,
                }
            )

    return connection_between_relationships


def is_relationship_connected(
    entity_index, relationship_index, connection_between_relationships
):
    return any(
        (
            connection["entity_1"] == entity_index
            or connection["entity_2"] == entity_index
        )
        and (
            connection["relationship_1"] == relationship_index
            or connection["relationship_2"] == relationship_index
        )
        for connection in connection_between_relationships
    )


def search_connection(entity, relationship, entities, img):
    height, width, _ = img.shape
    displacement_measure = int(
        (relationship["box"]["xmax"] - relationship["box"]["xmin"]) / 2
    )
    current_location = relationship["box"].copy()
    movement_direction = get_first_movement(
        entity_box=entity["box"], relationship_box=relationship["box"]
    )

    entity_index_found = None
    relationship_index_found = None
    move_counter = 0

    current_location = first_move_towards(
        current_location=current_location,
        movement_direction=movement_direction,
        height=height,
        width=width,
    )

    while move_counter < 250:

        midpoint = get_next_midpoint(current_location, movement_direction)

        entity_index_found, relationship_index_found = find_relationship(
            midpoint=midpoint, entities=entities
        )

        if entity_index_found != None:
            break

        line_type = recognize_line(
            img[
                current_location["ymin"] : current_location["ymax"],
                current_location["xmin"] : current_location["xmax"],
            ]
        )

        movement_direction = get_next_movement_direction(
            line_type=line_type, actual_movement_direction=movement_direction
        )

        if movement_direction == None:
            break

        current_location = move_towards(
            current_location=current_location,
            displacement_measure=displacement_measure,
            direction=movement_direction,
            height=height,
            width=width,
        )

        move_counter += 1

    return entity_index_found, relationship_index_found


def get_first_movement(entity_box, relationship_box):
    midpoint = get_next_midpoint(relationship_box, DIRECTION.UP)
    if is_midpoint_inside_box(midpoint=midpoint, box=entity_box):
        return DIRECTION.DOWN
    midpoint = get_next_midpoint(relationship_box, DIRECTION.DOWN)
    if is_midpoint_inside_box(midpoint=midpoint, box=entity_box):
        return DIRECTION.UP
    midpoint = get_next_midpoint(relationship_box, DIRECTION.RIGHT)
    if is_midpoint_inside_box(midpoint=midpoint, box=entity_box):
        return DIRECTION.LEFT
    return DIRECTION.RIGHT


def first_move_towards(current_location, movement_direction, height, width):
    match movement_direction:
        case DIRECTION.UP | DIRECTION.DOWN:
            displacement = current_location["ymax"] - current_location["ymin"]
            return move_towards(
                current_location=current_location,
                displacement_measure=displacement,
                direction=movement_direction,
                height=height,
                width=width,
            )

        case DIRECTION.RIGHT | DIRECTION.LEFT:
            displacement = current_location["xmax"] - current_location["xmin"]
            return move_towards(
                current_location=current_location,
                displacement_measure=displacement,
                direction=movement_direction,
                height=height,
                width=width,
            )


def get_next_midpoint(box, direction):
    # + y: go DOWN
    # + x: go RIGHT
    match direction:
        case DIRECTION.UP:
            x = box["xmin"] + int((box["xmax"] - box["xmin"]) / 2)
            y = box["ymin"] - int((box["ymax"] - box["ymin"]) / 2)
            return {"x": x, "y": y}
        case DIRECTION.DOWN:
            x = box["xmin"] + int((box["xmax"] - box["xmin"]) / 2)
            y = box["ymax"] + int((box["ymax"] - box["ymin"]) / 2)
            return {"x": x, "y": y}
        case DIRECTION.RIGHT:
            x = box["xmax"] + int((box["xmax"] - box["xmin"]) / 2)
            y = box["ymin"] + int((box["ymax"] - box["ymin"]) / 2)
            return {"x": x, "y": y}
        case DIRECTION.LEFT:
            x = box["xmin"] - int((box["xmax"] - box["xmin"]) / 2)
            y = box["ymin"] + int((box["ymax"] - box["ymin"]) / 2)
            return {"x": x, "y": y}


def move_towards(current_location, displacement_measure, direction, height, width):
    match direction:
        case DIRECTION.UP:
            if current_location["ymin"] - displacement_measure < 0:
                return {
                    "xmin": current_location["xmin"],
                    "xmax": current_location["xmax"],
                    "ymin": 0,
                    "ymax": current_location["ymax"] - current_location["ymin"],
                }
            else:
                return {
                    "xmin": current_location["xmin"],
                    "xmax": current_location["xmax"],
                    "ymin": current_location["ymin"] - displacement_measure,
                    "ymax": current_location["ymax"] - displacement_measure,
                }

        case DIRECTION.DOWN:
            if current_location["ymax"] + displacement_measure > height:
                return {
                    "xmin": current_location["xmin"],
                    "xmax": current_location["xmax"],
                    "ymin": height
                    - (current_location["ymax"] - current_location["ymin"]),
                    "ymax": height,
                }
            else:
                return {
                    "xmin": current_location["xmin"],
                    "xmax": current_location["xmax"],
                    "ymin": current_location["ymin"] + displacement_measure,
                    "ymax": current_location["ymax"] + displacement_measure,
                }

        case DIRECTION.RIGHT:
            if current_location["xmax"] + displacement_measure > width:
                return {
                    "xmin": width
                    - (current_location["xmax"] - current_location["xmin"]),
                    "xmax": width,
                    "ymin": current_location["ymin"],
                    "ymax": current_location["ymax"],
                }
            else:
                return {
                    "xmin": current_location["xmin"] + displacement_measure,
                    "xmax": current_location["xmax"] + displacement_measure,
                    "ymin": current_location["ymin"],
                    "ymax": current_location["ymax"],
                }

        case DIRECTION.LEFT:
            if current_location["xmin"] - displacement_measure < 0:
                return {
                    "xmin": 0,
                    "xmax": current_location["xmax"] - current_location["xmin"],
                    "ymin": current_location["ymin"],
                    "ymax": current_location["ymax"],
                }
            else:
                return {
                    "xmin": current_location["xmin"] - displacement_measure,
                    "xmax": current_location["xmax"] - displacement_measure,
                    "ymin": current_location["ymin"],
                    "ymax": current_location["ymax"],
                }


def find_relationship(midpoint, entities):
    for entity in entities:
        for relationship in entity["relationships"]:
            if is_midpoint_inside_box(midpoint=midpoint, box=relationship["box"]):
                return entity["index"], relationship["index"]
    return None, None


def get_next_movement_direction(line_type, actual_movement_direction):
    match line_type:
        case LINE_CLASSES.CROSSING.value:
            return actual_movement_direction
        case LINE_CLASSES.HORIZONTAL.value:
            return actual_movement_direction
        case LINE_CLASSES.BOTTOM_LEFT.value:
            if actual_movement_direction == DIRECTION.UP:
                return DIRECTION.LEFT
            elif actual_movement_direction == DIRECTION.RIGHT:
                return DIRECTION.DOWN
            else:
                return actual_movement_direction
        case LINE_CLASSES.TOP_LEFT.value:
            if actual_movement_direction == DIRECTION.DOWN:
                return DIRECTION.LEFT
            elif actual_movement_direction == DIRECTION.RIGHT:
                return DIRECTION.UP
            else:
                return actual_movement_direction
        case LINE_CLASSES.NONE.value:
            return None
        case LINE_CLASSES.BOTTOM_RIGHT.value:
            if actual_movement_direction == DIRECTION.UP:
                return DIRECTION.RIGHT
            elif actual_movement_direction == DIRECTION.LEFT:
                return DIRECTION.DOWN
            else:
                return actual_movement_direction
        case LINE_CLASSES.TOP_RIGHT.value:
            if actual_movement_direction == DIRECTION.DOWN:
                return DIRECTION.RIGHT
            elif actual_movement_direction == DIRECTION.LEFT:
                return DIRECTION.UP
            else:
                return actual_movement_direction
        case LINE_CLASSES.VERTICAL.value:
            return actual_movement_direction
