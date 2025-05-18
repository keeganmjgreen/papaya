from __future__ import annotations

import abc
import dataclasses
import datetime as dt
from enum import Enum
import typing
import warnings
from typing import Any, Iterator, Optional, Type

import pandas as pd
import pandera.pandas as pa
from pandera.backends.base import BaseSchemaBackend
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.engines.pandas_engine import DateTime

from utils import get_exactly_one


class ObjectsDataframeBase[T](pd.DataFrame, abc.ABC):
    @property
    def _dataframe_objects_class(self) -> typing.Type[T]:
        return typing.get_args(self.__orig_class__)[0]

    def validate(self) -> None:
        schema = self._get_dataframe_schema()
        for field_name, dtype in schema.dtypes.items():
            if self[field_name].isna().all():  # Also True if DataFrame is empty.
                try:
                    self[field_name] = self[field_name].astype(str(dtype))
                except TypeError:
                    pass
            if isinstance(dtype, DateTime):
                if (
                    dtype.tz is not None
                    and isinstance(self[field_name].dtype, pd.DatetimeTZDtype)
                    and dtype.tz != self[field_name].dtype.tz
                ):
                    warnings.warn(
                        f"Converting column {field_name} from `tz={self[field_name].dtype.tz}` to "
                        f"`tz={dtype.tz}`."
                    )
                    self[field_name] = self[field_name].dt.tz_convert(dtype.tz)
        schema.validate(pd.DataFrame(self))

    def _get_dataframe_schema(self) -> DataframeSchema:
        return DataframeSchema(
            {
                f.name: self._get_column_schema(
                    f.name,
                    *self._dataframe_objects_class._process_type_annotation(f.type),
                )
                for f in dataclasses.fields(self._dataframe_objects_class)
            }
        )

    def _get_column_schema(
        self,
        field_name: str,
        annotated_type: type,
        nullable: bool,
        annotated_with: type | None = None,
        **kwargs,
    ) -> pa.Column:
        if annotated_type is bool and nullable:
            if getattr(self, "store_nullable_bools_as_objects", False):
                self[field_name] = self[field_name].astype(object)
                return pa.Column(object, nullable=True, **kwargs)
            else:
                self[field_name] = self[field_name].astype(pd.BooleanDtype())
                warnings.warn(
                    "Using Pandas' experimental `BooleanDtype` for nullable bool field. "
                    "To avoid this, set "
                    "`<ObjectsBackingDataframe instance>.store_nullable_bools_as_objects = True`."
                )
                return pa.Column(pd.BooleanDtype(), nullable=True, **kwargs)

        if annotated_type is dt.date:
            return pa.Column(object, nullable=nullable, **kwargs)

        if issubclass(annotated_type, Enum):
            store_enum_members_as = getattr(self, "store_enum_members_as", "members")
            if store_enum_members_as == "members":
                return pa.Column(object, nullable=nullable, **kwargs)
            elif store_enum_members_as == "names":
                self[field_name] = (
                    self[field_name]
                    .apply(lambda x: x.name if not isinstance(x, str) else x)
                    .astype(str)
                )
                return pa.Column(str, nullable=nullable, **kwargs)
            elif store_enum_members_as == "values":
                member_values = {
                    member.value for member in annotated_type.__members__.values()
                }
                try:
                    enum_value_type = get_exactly_one(
                        set(type(x) for x in member_values)
                    )
                except ValueError:
                    warnings.warn(
                        f"The member values of enum `{annotated_type.__name__}` are not all of the "
                        "same type. Using `object`."
                    )
                    enum_value_type = object
                return self._get_column_schema(
                    field_name,
                    enum_value_type,
                    nullable,
                    checks=(lambda x: x in member_values),
                )

        if annotated_type is int and nullable:
            if getattr(self, "store_nullable_ints_as_floats", False):
                self[field_name] = self[field_name].astype(float)
                return pa.Column(float, nullable=True, **kwargs)
            else:
                self[field_name] = self[field_name].astype(pd.Int64Dtype())
                warnings.warn(
                    "Using Pandas' experimental `Int64Dtype` for nullable integer field. "
                    "To avoid this, set "
                    "`<ObjectsBackingDataframe instance>.store_nullable_ints_as_floats = True`."
                )
                return pa.Column(pd.Int64Dtype(), nullable=True, **kwargs)

        if annotated_type is pd.Timestamp or annotated_type is dt.datetime:
            annotated_with: DateTime
            return pa.Column(annotated_with, nullable=nullable, **kwargs)

        if annotated_type is str:
            return pa.Column(object, nullable=nullable, **kwargs)

        try:
            return pa.Column(annotated_type, nullable=nullable, **kwargs)
        except TypeError:
            return pa.Column(object, nullable=nullable, **kwargs)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError


class DataframeSchema(pa.DataFrameSchema):
    @classmethod
    @classmethod
    def get_backend(
        cls,
        check_obj: Optional[Any] = None,
        check_type: Optional[Type] = None,
    ) -> BaseSchemaBackend:
        return DataFrameSchemaBackend()
