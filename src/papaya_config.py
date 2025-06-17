import dataclasses
from typing import Literal


@dataclasses.dataclass
class PapayaConfig:
    store_nullable_bools_as_objects: bool = False
    store_dates_as_timestamps: bool = False
    store_enum_members_as: Literal["members", "names", "values"] = "members"
    store_nullable_ints_as_floats: bool = False

    set_index: bool | Literal["auto"] = False
    """
    If fields are annotated with `DataframeIndex`:
        True: Require that such fields are in the columns. They will be moved to the
            [Multi]Index.
        False: Require that such fields are already in the [Multi]Index.
        "auto": Move such fields to the [Multi]Index if they are all in the columns.
    """
