import base64

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .ERmodel import get_ERmodel

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")


class ERmodelBody(BaseModel):
    img: str = Field(
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQ..."]
    )


class BoxResponse(BaseModel):
    xmin: int = Field(examples=["43"])
    xmax: int = Field(examples=["145"])
    ymin: int = Field(examples=["224"])
    ymax: int = Field(examples=["322"])


class RelationshipResponse(BaseModel):
    index: int = Field(examples=["0"])
    class_relationship: str = Field(alias="class", examples=["01", "11", "0M", "1M"])
    box: BoxResponse


class EntityResponse(BaseModel):
    index: int = Field(examples=["0"])
    class_entity: str = Field(alias="class", examples=["ENTITY"])
    box: BoxResponse
    name: str = Field(examples=["Entity name"])
    attributes: list[str] = Field(examples=[["Attribute 1", "Attribute 2"]])
    relationships: list[RelationshipResponse]


class ConnectionResponse(BaseModel):
    entity_1: int = Field(examples=["0"])
    relationship_1: int = Field(examples=["0"])
    entity_2: int = Field(examples=["1"])
    relationship_2: int = Field(examples=["0"])


class ERmodelResponse(BaseModel):
    entities: list[EntityResponse]
    connections: list[ConnectionResponse]


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ermodel", response_model=ERmodelResponse)
def analize_ERmodel(ermodelBody: ERmodelBody):

    if ermodelBody.img.startswith("data:image/"):
        imgBase64 = ermodelBody.img.split(",")[1]
    else:
        imgBase64 = ermodelBody.img

    decoded_data = base64.b64decode(imgBase64)
    nparr = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = get_ERmodel(img)
    return result
