import dataclasses
import datetime as dt
import types
import typing
from functools import partial
from typing import Any, Iterator

import pandas as pd
from pandera.engines.pandas_engine import DateTime
from typing_extensions import Literal

from objects_dataframe_base import ObjectsDataframeBase
from papaya_types import find_papaya_type
from utils import get_exactly_one


class ObjectsBackingDataframe[T](ObjectsDataframeBase):

    def __iter__(self) -> Iterator[T]:
        for df_key, _ in super().iterrows():
            dataframe_backed_class = self._dataframe_objects_class
            dataframe_backed_intance = dataframe_backed_class(
                __df=self, __df_key=df_key
            )
            yield dataframe_backed_intance


def dataframe_backed_object(cls):
    cls.__df = None
    cls.__df_key = None

    original_constructor = cls.__init__

    def wrapped_constructor(self, *args, **kwargs):
        if kwargs.get("__df") is not None and kwargs.get("__df_key") is not None:
            original_constructor(
                self, **{f.name: None for f in dataclasses.fields(cls)}
            )
            self.__df = kwargs["__df"]
            self.__df_key = kwargs["__df_key"]
        else:
            original_constructor(self, *args, **kwargs)

    cls.__init__ = wrapped_constructor

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
                    "`Optional[<type>]`, i.e., `Union[<type>, None]` or "
                    "`<type> | None`."
                ) from None
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
            args = typing.get_args(type_annotation)
            type_annotation, annotated_with = args[0], args[1]
        else:
            annotated_with = None

        if type_annotation is pd.Timestamp or type_annotation is dt.datetime:
            if not isinstance(annotated_with, DateTime):
                raise TypeError(
                    "`pandas.Timestamp` or `datetime.datetime` alone are insufficient "
                    "type annotations; they do not enforce time zones or even whether "
                    "a timestamp/datetime is time-zone-aware. Instead, use:\n"
                    "```python\n"
                    "from typing import Annotated\n"
                    "from zoneinfo import ZoneInfo\n"
                    "\n"
                    "from pandera.engines.pandas_engine import DateTime\n"
                    "\n"
                    f"Annotated[{type_annotation.__name__}, DateTime(tz=ZoneInfo(...))]\n"  # noqa: E501
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
            if field_name in self.__df.columns:
                value = self.__df.loc[self.__df_key, field_name]
            elif isinstance(self.__df.index, pd.MultiIndex):
                value = self.__df.loc[self.__df_key].name[
                    list(self.__df.index.names).index(field_name)
                ]
            elif self.__df.index.name == field_name:
                value = self.__df_key
            fields = {f.name: f for f in dataclasses.fields(self)}
            field = fields[field_name]
            papaya_type = find_papaya_type(
                field_name,
                *type(self)._process_type_annotation(field.type),
                config=self.__df.papaya_config,
            )
            return papaya_type.process_getter_value(value)
        else:
            return getattr(self, f"__{field_name}")

    def fset(self, value: Any, field_name: str) -> None:
        if self.__df is not None and self.__df_key is not None:
            if field_name in self.__df.index.names:
                raise SettingOnIndexLevelError
            fields = {f.name: f for f in dataclasses.fields(self)}
            field = fields[field_name]
            papaya_type = find_papaya_type(
                field_name,
                *type(self)._process_type_annotation(field.type),
                config=self.__df.papaya_config,
            )
            self.__df.at[self.__df_key, field_name] = papaya_type.process_setter_value(
                value
            )
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


class SettingOnIndexLevelError(Exception):
    pass
