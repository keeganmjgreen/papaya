from __future__ import annotations

import abc
import copy
import dataclasses
import typing
import warnings
from typing import Any, Iterator, Optional, Type, _GenericAlias

import pandas as pd
import pandera.pandas as pa
from pandera import DataType, MultiIndex
from pandera.backends.base import BaseSchemaBackend
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.engines.pandas_engine import DateTime

from papaya_config import PapayaConfig
from papaya_types import find_papaya_type
from utils import get_exactly_one

# The following is copied from module `pandera.typing.common`, with `DataFrameBase`
# replaced with `ObjectsDataframeBase`. Refer to `ObjectsDataframeBase.__setattr__` for
# more information.

__orig_generic_alias_call = copy.copy(_GenericAlias.__call__)


def __patched_generic_alias_call(self, *args, **kwargs):
    """
    Patched implementation of _GenericAlias.__call__ so that validation errors
    can be raised when instantiating an instance of ObjectsBackingDataframe generics,
    e.g. ObjectsBackingDataframe[A](data).
    """
    if ObjectsDataframeBase not in self.__origin__.__bases__:
        return __orig_generic_alias_call(self, *args, **kwargs)

    if not self._inst:
        raise TypeError(
            f"Type {self._name} cannot be instantiated; "
            f"use {self.__origin__.__name__}() instead"
        )
    result = self.__origin__(*args, **kwargs)
    try:
        result.__orig_class__ = self
    # Limit the patched behavior to subset of exception types
    except (
        TypeError,
        ValueError,
        pa.errors.SchemaError,
        pa.errors.SchemaInitError,
        pa.errors.SchemaDefinitionError,
    ):
        raise
    # In python 3.11.9, all exceptions when setting attributes when defining
    # _GenericAlias subclasses are caught and ignored.
    except Exception:  # pylint: disable=broad-except
        pass
    return result


_GenericAlias.__call__ = __patched_generic_alias_call

# The preceeding is copied from module `pandera.typing.common`.


class ObjectsDataframeBase[T](pd.DataFrame, abc.ABC):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        # We want to validate instances of `ObjectsDataframeBase` upon instantiation.
        # The schema against which it is validated is determined by the dataclass held
        # by TypeVar `T`. This is accessible through property
        # `_dataframe_objects_class`, which uses attribute `__orig_class__`.
        # `__orig_class__` is set by Python and is an implementation detail rather than
        # a supported feature, but we still want to use it. However, since Python 3.7,
        # it is set *after* on an instance after it is instantiated, meaning that it is
        # unavailable for us to use to validate the DataFrame in `__init__`. As a
        # workaround, detect when `__orig_class__` is set by Python and in that moment
        # use it to validate the DataFrame. This is also what Pandera does.
        if name == "__orig_class__":
            self.validate()
            # In at least some Python versions, any errors encountered by Python when
            # setting `__orig_class__` are caught, preventing us from raising from
            # `.validate` as desired. `__patched_generic_alias_call` overrides this
            # Python behavior.

    def validate(self) -> None:
        schema = self._get_dataframe_schema()

        if len(self._index_fields) > 0:
            self._prevalidate_index()

        self._process_fields(schema)

        schema.validate(pd.DataFrame(self))

    def _get_dataframe_schema(self) -> DataframeSchema:
        field_to_schema_dict = {
            f: find_papaya_type(
                f.name,
                *self._dataframe_objects_class._process_type_annotation(f.type),
                config=self.papaya_config,
            )
            .validator(self)
            .__dict__
            for f in self._fields
        }
        for f, s in field_to_schema_dict.items():
            s["dtype"] = s.pop("_dtype")
            field_to_schema_dict[f] = s

        if len(self._index_fields) == 0:
            index_schema = None
        else:
            index_levels = [
                pa.Index(**s)
                for f, s in field_to_schema_dict.items()
                if f in self._index_fields
            ]
            if len(index_levels) == 1:
                index_schema = get_exactly_one(index_levels)
            else:
                index_schema = pa.MultiIndex(index_levels)

        return DataframeSchema(
            {
                f.name: pa.Column(**s)
                for f, s in field_to_schema_dict.items()
                if f not in self._index_fields
            },
            index=index_schema,
        )

    def _prevalidate_index(self):
        index_fields_in_columns = [f.name in self.columns for f in self._index_fields]
        fields_in_index = [f.name in self.index.names for f in self._fields]
        index_fields_equals_index = self.index.names == [
            f.name for f in self._index_fields
        ]
        if self.papaya_config.set_index is True:
            if any(fields_in_index):
                raise ValueError(
                    "When `PapayaConfig.set_index` is True, there must be no "
                    "fields in the index."
                )
            if not all(index_fields_in_columns):
                raise ValueError(
                    "When `PapayaConfig.set_index` is True, all index fields must "
                    "be in the columns."
                )
            self.set_index([f.name for f in self._index_fields], inplace=True)
        elif self.papaya_config.set_index is False:
            if not index_fields_equals_index:
                raise ValueError(
                    "When `PapayaConfig.set_index` is False, the index fields must "
                    "exactly match the index levels."
                )
            if any(index_fields_in_columns):
                raise ValueError(
                    "When `PapayaConfig.set_index` is False, there must be no "
                    "index fields in the columns."
                )
        elif self.papaya_config.set_index == "auto":
            if index_fields_in_columns is index_fields_equals_index:
                raise ValueError(
                    'When `PapayaConfig.set_index` is "auto", the index fields '
                    "must either be in the columns or exactly match the index "
                    "levels."
                )
            if index_fields_in_columns:
                self.set_index([f.name for f in self._index_fields], inplace=True)

        else:
            raise ValueError("Unexpected value for `PapayaConfig.set_index`.")

    def _process_fields(self, schema: DataframeSchema) -> None:
        for field in self._non_index_fields:
            dtype = schema.dtypes[field.name]
            if field.name in self.columns:
                self[field.name] = self._process_field(self[field.name], dtype)
        for field in self._index_fields:
            dtype = (
                schema.index.dtypes[field.name]
                if isinstance(schema.index, MultiIndex)
                else schema.index.dtype
            )
            if field.name in self.index.names:
                if isinstance(self.index, pd.MultiIndex):
                    self.index = self.index.set_levels(
                        self._process_field(
                            self.index.get_level_values(field.name), dtype
                        ),
                        level=field.name,
                    )
                else:
                    self.index = self._process_field(self.index, dtype)

    def _process_field(
        self, array: pd.Series | pd.Index, dtype: DataType
    ) -> pd.Series | pd.Index:
        if array.isna().all():  # Also True if DataFrame is empty.
            try:
                array = array.astype(str(dtype))
            except TypeError:
                pass  # `.validate` will handle.
        if isinstance(dtype, DateTime):
            if (
                dtype.tz is not None
                and isinstance(array.dtype, pd.DatetimeTZDtype)
                and dtype.tz != array.dtype.tz
            ):
                field_name = array.name
                warnings.warn(
                    f"Converting {field_name} from `tz={array.dtype.tz}` to "
                    f"`tz={dtype.tz}`.",
                    stacklevel=2,
                )
                if isinstance(array, pd.Series):
                    array = array.dt.tz_convert(dtype.tz)
                elif isinstance(array, pd.Index):
                    array = array.tz_convert(dtype.tz)
        return array

    @property
    def papaya_config(self) -> PapayaConfig:
        return getattr(self._dataframe_objects_class, "papaya_config", PapayaConfig())

    @property
    def _dataframe_objects_class(self) -> typing.Type[T]:
        return typing.get_args(self.__orig_class__)[0]

    @property
    def _fields(self) -> list[dataclasses.Field]:
        return list(dataclasses.fields(self._dataframe_objects_class))

    @property
    def _index_fields(self) -> list[dataclasses.Field]:
        return [
            f
            for f in self._fields
            if isinstance(f.type, typing._AnnotatedAlias)
            and DataframeIndex in typing.get_args(f.type)[1:]
        ]

    @property
    def _non_index_fields(self) -> list[dataclasses.Field]:
        return [f for f in self._fields if f not in self._index_fields]

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError


class DataframeIndex:
    pass


class DataframeSchema(pa.DataFrameSchema):
    @classmethod
    @classmethod
    def get_backend(
        cls,
        check_obj: Optional[Any] = None,
        check_type: Optional[Type] = None,
    ) -> BaseSchemaBackend:
        return DataFrameSchemaBackend()
