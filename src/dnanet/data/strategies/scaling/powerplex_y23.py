from dnanet.data.strategies.scaling.kit import PPY23_KIT, STRKit
from dnanet.data.strategies.scaling.size_standard import WEN_ILS
from dnanet.data.strategies.scaling.powerplex_fusion_6c import PowerPlexFusion6CStrategy


# TODO: should this be a child of PowerPlexFusion6CStrategy?
class PowerplexY23(PowerPlexFusion6CStrategy):
    def __init__(self):
        kit = PPY23_KIT
        super().__init__(kit=kit)

    def marker_name_to_dye_idx(self) -> dict[str, int]:
        return {
            'DYS576': 0,
            'DYS389I': 0,
            'DYS448': 0,
            'DYS389II': 0,
            'DYS19': 0,
            'DYS391': 1,
            'DYS481': 1,
            'DYS549': 1,
            'DYS533': 1,
            'DYS438': 1,
            'DYS437': 1,
            'DYS570': 2,
            'DYS635': 2,
            'DYS390': 2,
            'DYS439': 2,
            'DYS392': 2,
            'DYS643': 2,
            'DYS393': 3,
            'DYS458': 3,
            'DYS385': 3,
            'DYS456': 3,
            'YGATAH4': 3,
        }

    def dye_channel_colors(self) -> list[str]:
        return ['blue', 'green', 'black', 'red', 'orange']
