from dnanet.data.strategies.kit import STRKit
from dnanet.data.strategies.scaling.powerplex_fusion_6c import PowerPlexFusion6CStrategy
from dnanet.data.strategies.size_standard import WEN_ILS


class PowerplexY23(PowerPlexFusion6CStrategy):
    def __init__(self, panel_path: str = None):
        kit = STRKit(
            name="POWERPLEX_Y23",
            size_standard=WEN_ILS,
            panel=None, ## TODO: add panel
            num_dyes=5,
            hid_dye_mapping=self._HID_DYE_MAPPING,
            panel_path=panel_path,
            description="POWERPLEX_Y23 kit using WEN_ILS size standard.",
            hid_file_data_columns_raw=["DATA_1","DATA_2", "DATA_3","DATA_4", "DATA_105"],
            hid_file_data_columns_analyzed=["DATA_9","DATA_10", "DATA_11","DATA_12", "DATA_205"]
        )
        super().__init__(kit=kit, panel_path=panel_path)

    def marker_name_to_dye_idx(self) -> dict[str, int]:
         return {
            "DYS576": 0, "DYS389I": 0, "DYS448": 0, "DYS389II": 0, "DYS19": 0,
            "DYS391": 1, "DYS481": 1, "DYS549": 1, "DYS533": 1, "DYS438": 1, "DYS437": 1,
            "DYS570": 2, "DYS635": 2, "DYS390": 2, "DYS439": 2, "DYS392": 2, "DYS643": 2,
            "DYS393": 3, "DYS458": 3, "DYS385": 3, "DYS456": 3, "YGATAH4": 3,
        }

    def dye_channel_colors(self) -> list[str]:
        return ["blue", "green", "black", "red", "orange"]
