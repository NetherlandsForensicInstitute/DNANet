from typing import Annotated, List, Union
import numpy as np
from pydantic import BaseModel, Field
import pydantic_numpy.typing as pnp


class AlleleAnnotation(BaseModel):
    annotation: List[str]

class PixelAnnotation(BaseModel):
    annotation: pnp.Np2DArrayFp64
    
Annotation = Annotated[Union[AlleleAnnotation, PixelAnnotation], Field(discriminator="type")]


class AllelePrediction(AlleleAnnotation):
    pass

class PixelPrediction(PixelAnnotation):
    pass
    
Prediction = Annotated[Union[AllelePrediction, PixelPrediction], Field(discriminator="type")]