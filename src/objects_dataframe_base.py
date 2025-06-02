from __future__ import annotations

import abc
import dataclasses
import typing
import warnings
from typing import Any, Iterator, Optional, Type

import pandas as pd
import pandera.pandas as pa
from pandera.backends.base import BaseSchemaBackend
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.engines.pandas_engine import DateTime

from pya_types import PyaTypesConfig, find_pya_type


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
                        f"Converting column {field_name} from "
                        f"`tz={self[field_name].dtype.tz}` to `tz={dtype.tz}`.",
                        stacklevel=2,
                    )
                    self[field_name] = self[field_name].dt.tz_convert(dtype.tz)
        schema.validate(pd.DataFrame(self))

    def _get_dataframe_schema(self) -> DataframeSchema:
        return DataframeSchema(
            {
                f.name: find_pya_type(
                    f.name,
                    *self._dataframe_objects_class._process_type_annotation(f.type),
                    config=self.pya_types_config,
                ).validator(self)
                for f in dataclasses.fields(self._dataframe_objects_class)
            }
        )

    @property
    def pya_types_config(self) -> PyaTypesConfig:
        return PyaTypesConfig(
            store_nullable_bools_as_objects=getattr(
                self, "store_nullable_bools_as_objects", False
            ),
            store_dates_as_timestamps=getattr(self, "store_dates_as_timestamps", False),
            store_enum_members_as=getattr(self, "store_enum_members_as", "members"),
            store_nullable_ints_as_floats=getattr(
                self, "store_nullable_ints_as_floats", False
            ),
        )

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
