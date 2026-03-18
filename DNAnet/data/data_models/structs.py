from typing import Annotated, Dict, List, Union

from pydantic import BaseModel, Field, model_validator
import pydantic_numpy.typing as pnp
from DNAnet.data.data_models.dna_models import Marker


class AlleleAnnotation(BaseModel):
    annotation: List[Marker]

    @model_validator(mode="before")
    @classmethod
    def validate_annotation_input(cls, data: dict):
        if isinstance(data, dict):
            value = data.get("annotation")

            if isinstance(value, list) and not isinstance(value[0], Marker):
                # Merge markers
                markers: Dict[str, Marker] = {}

                for sample in value:
                    for sample_marker in sample:
                        if (marker := markers.get(sample_marker.name)) is None:
                            markers[sample_marker.name] = sample_marker
                        else:
                            marker.alleles.extend(sample_marker.alleles)
                return {"annotation": list(markers.values())}
        return data


class ScanpointAnnotation(BaseModel):
    annotation: pnp.Np2DArrayInt8 # annotations only store the class index, so int8 is sufficient

Annotation = Annotated[Union[AlleleAnnotation, ScanpointAnnotation], Field(discriminator="type")]


class AllelePrediction(AlleleAnnotation):
    pass

class ScanpointPrediction(ScanpointAnnotation):
    pass

Prediction = Annotated[Union[AllelePrediction, ScanpointPrediction], Field(discriminator="type")]
