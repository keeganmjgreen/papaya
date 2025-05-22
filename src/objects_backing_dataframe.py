import dataclasses
import datetime as dt
import types
import typing
from functools import partial
from typing import Any, Iterator
from typing_extensions import Literal

import pandas as pd
from pandera.engines.pandas_engine import DateTime

from objects_dataframe_base import ObjectsDataframeBase
from pya_types import find_pya_type
from utils import get_exactly_one


class ObjectsBackingDataframe[T](ObjectsDataframeBase):
    def __iter__(self) -> Iterator[T]:
        self.validate()
        for df_key, _ in super().iterrows():
            dataframe_backed_class = self._dataframe_objects_class
            dataframe_backed_intance = dataframe_backed_class(
                __df=self, __df_key=df_key
            )
            yield dataframe_backed_intance


def dataframe_backed_object(cls):
    cls.__df = None
    cls.__df_key = None

    init = cls.__init__

    def __init__(self, *args, **kwargs):
        if kwargs.get("__df") is not None and kwargs.get("__df_key") is not None:
            init(self, **{f.name: None for f in dataclasses.fields(cls)})
            self.__df = kwargs["__df"]
            self.__df_key = kwargs["__df_key"]
        else:
            init(self, *args, **kwargs)

    cls.__init__ = __init__

    def _process_type_annotation(type_annotation: type) -> tuple[type, type, bool]:
        if isinstance(type_annotation, types.UnionType) or isinstance(
            type_annotation, typing._UnionGenericAlias
        ):
            unioned_types = list(typing.get_args(type_annotation))
            if types.NoneType in unioned_types:
                nullable = True
                unioned_types.remove(types.NoneType)
            else:
                nullable = False
            try:
                type_annotation = get_exactly_one(unioned_types)
            except ValueError:
                raise TypeError(
                    "Union types not supported unless being used as an alias for "
                    "`Optional[<type>]`, i.e., `Union[<type>, None]` or `<type> | None`."
                )
        else:
            nullable = False

        if isinstance(type_annotation, typing._LiteralGenericAlias):
            literal_values = list(typing.get_args(type_annotation))
            if None in literal_values:
                nullable = True
                literal_values.remove(None)
            else:
                nullable = False
            type_annotation = Literal[*literal_values]

        if isinstance(type_annotation, typing._AnnotatedAlias):
            type_annotation, annotated_with = typing.get_args(type_annotation)
        else:
            annotated_with = False

        if type_annotation is pd.Timestamp or type_annotation is dt.datetime:
            if not isinstance(annotated_with, DateTime):
                raise TypeError(
                    "`pandas.Timestamp` or `datetime.datetime` alone are insufficient type "
                    "annotations; they do not enforce time zones or even whether a "
                    "timestamp/datetime is time-zone-aware. Instead, use:\n"
                    "```python\n"
                    "from typing import Annotated\n"
                    "from zoneinfo import ZoneInfo\n"
                    "\n"
                    "from pandera.engines.pandas_engine import DateTime\n"
                    "\n"
                    f"Annotated[{type_annotation.__name__}, DateTime(tz=ZoneInfo(...))]\n"
                    "```\n"
                    "Or, for time-zone-naiveness (not advisable), use:\n"
                    "```python\n"
                    f"Annotated[{type_annotation.__name__}, DateTime(tz=None)]\n"
                    "```"
                )

        return type_annotation, annotated_with, nullable

    for field in dataclasses.fields(cls):
        _process_type_annotation(field.type)
    cls._process_type_annotation = _process_type_annotation

    def fget(self, field_name: str) -> Any:
        if self.__df is not None and self.__df_key is not None:
            value = self.__df.loc[self.__df_key, field_name]
            fields = {f.name: f for f in dataclasses.fields(self)}
            field = fields[field_name]
            pya_type = find_pya_type(
                field_name,
                *type(self)._process_type_annotation(field.type),
                config=self.__df.pya_types_config,
            )
            return pya_type.process_getter_value(value)
        else:
            return getattr(self, f"__{field_name}")

    def fset(self, value: Any, field_name: str) -> None:
        if self.__df is not None and self.__df_key is not None:
            self.__df.loc[self.__df_key, field_name] = value
        else:
            setattr(self, f"__{field_name}", value)

    for field_name in [f.name for f in dataclasses.fields(cls)]:
        setattr(
            cls,
            field_name,
            property(
                fget=partial(fget, field_name=field_name),
                fset=partial(fset, field_name=field_name),
            ),
        )

    return cls
