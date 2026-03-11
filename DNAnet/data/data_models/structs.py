from pathlib import Path
from typing import Annotated, List, Union

from pydantic import BaseModel, Field, model_validator
import pydantic_numpy.typing as pnp
from DNAnet.data.data_models.dna_models import Marker
from DNAnet.data.strategies.strategy_registry import StrategyRegistry


class AlleleAnnotation(BaseModel):
    annotation: List[Marker]
    
    @model_validator(mode="before")
    @classmethod
    def validate_annotation_input(cls, data: dict):
        if isinstance(data, dict):
            value = data.get("annotation")
            
            if isinstance(value, (str, Path)):
                # Get the dataset strategy for reading annotations
                _dataset_strategy = StrategyRegistry.get_dataset()
                
                # Read the file
                _markers = _dataset_strategy.parse_annotation_file(
                    path=value,
                    sample_name=data.get("sample_name")
                )
                data["annotation"] = _markers
            
        return data

class PixelAnnotation(BaseModel):
    annotation: pnp.Np2DArrayFp64
    
Annotation = Annotated[Union[AlleleAnnotation, PixelAnnotation], Field(discriminator="type")]


class AllelePrediction(AlleleAnnotation):
    pass

class PixelPrediction(PixelAnnotation):
    pass
    
Prediction = Annotated[Union[AllelePrediction, PixelPrediction], Field(discriminator="type")]