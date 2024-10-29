import base64

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .ERmodel import get_ERmodel

app = FastAPI()


class ERmodelBody(BaseModel):
    img: str


@app.post("/der")
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
