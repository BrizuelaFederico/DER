import cv2

from .AI.ai_utils import crop_image, expand_box
from .AI.entity.ai_entity import get_entities
from .AI.relationship.ai_relationship import get_relationships
from .AI.text.ai_entity_text import get_entity_text
from .connection_between_relationships import get_connection_between_relationships


def get_ERmodel(img):
    margin = get_margin(img=img)
    blank_margins = add_blank_margins(img=img, margin=margin)
    entities = get_entities(img=blank_margins)
    entities_with_name_and_atributtes = list(
        map(
            lambda entity: get_entity_name_and_attributes(
                entity=entity, img=blank_margins
            ),
            entities,
        )
    )
    entities_with_relationships = list(
        map(
            lambda entity: get_entity_relationships(entity=entity, img=blank_margins),
            entities_with_name_and_atributtes,
        )
    )

    connection_between_relationships = get_connection_between_relationships(
        entities=entities_with_relationships, img=blank_margins
    )

    entities_without_margin = remove_white_margin_from_boxes(
        entities=entities_with_relationships, margin=margin
    )

    return {
        "entities": entities_without_margin,
        "connections": connection_between_relationships,
    }


def get_margin(img):
    height, width, _ = img.shape
    return int((height + width) / 2 * 0.1)


# This is to be able to detect objects that are on the edge
def add_blank_margins(img, margin):
    return cv2.copyMakeBorder(
        img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, None, (255, 255, 255)
    )


def get_entity_name_and_attributes(entity, img):
    cropped_image = crop_image(box=entity["box"], img=img)
    name_and_attributes = get_entity_text(img=cropped_image)
    entity.update(name_and_attributes)
    return entity


def get_entity_relationships(entity, img):
    height, width, _ = img.shape
    expanded_box = expand_box(box=entity["box"], width=width, height=height)
    cropped_image = crop_image(box=expanded_box, img=img)
    relationships = get_relationships(img=cropped_image)

    real_region_relationships = list(
        map(
            lambda relationship: get_real_region_relationship(
                relationship=relationship, expanded_box=expanded_box
            ),
            relationships,
        )
    )

    entity["relationships"] = real_region_relationships
    return entity


def get_real_region_relationship(relationship, expanded_box):
    relationship["box"] = {
        "xmin": relationship["box"]["xmin"] + expanded_box["xmin"],
        "xmax": relationship["box"]["xmax"] + expanded_box["xmin"],
        "ymin": relationship["box"]["ymin"] + expanded_box["ymin"],
        "ymax": relationship["box"]["ymax"] + expanded_box["ymin"],
    }
    return relationship


def remove_white_margin_from_boxes(entities, margin):
    entities_without_margin = entities.copy()
    for entity in entities_without_margin:
        entity["box"] = {
            "xmin": max(0, entity["box"]["xmin"] - margin),
            "xmax": entity["box"]["xmax"] - margin,
            "ymin": max(0, entity["box"]["ymin"] - margin),
            "ymax": entity["box"]["ymax"] - margin,
        }
        for relationship in entity["relationships"]:
            relationship["box"] = {
                "xmin": max(0, relationship["box"]["xmin"] - margin),
                "xmax": relationship["box"]["xmax"] - margin,
                "ymin": max(0, relationship["box"]["ymin"] - margin),
                "ymax": relationship["box"]["ymax"] - margin,
            }
    return entities_without_margin
