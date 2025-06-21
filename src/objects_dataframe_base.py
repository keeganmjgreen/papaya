from __future__ import annotations

import abc
import copy
import dataclasses
import typing
import warnings
from typing import Any, Iterator, Optional, Type, _GenericAlias

import pandas as pd
import pandera.pandas as pa
from pandera.backends.base import BaseSchemaBackend
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.engines.pandas_engine import DateTime

from papaya_types import PapayaTypesConfig, find_papaya_type

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
                f.name: find_papaya_type(
                    f.name,
                    *self._dataframe_objects_class._process_type_annotation(f.type),
                    config=self.papaya_config,
                ).validator(self)
                for f in dataclasses.fields(self._dataframe_objects_class)
            }
        )

    @property
    def papaya_config(self) -> PapayaTypesConfig:
        return getattr(
            self._dataframe_objects_class, "papaya_config", PapayaTypesConfig()
        )

    @property
    def _dataframe_objects_class(self) -> typing.Type[T]:
        return typing.get_args(self.__orig_class__)[0]

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
