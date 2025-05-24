import abc
import dataclasses
import datetime as dt
import typing
import warnings
from enum import EnumType
from typing import Any, override

import pandas as pd
import pandera as pa
from pandera.engines.pandas_engine import DateTime
from typing_extensions import Literal

from utils import get_exactly_one


@dataclasses.dataclass
class PyaTypesConfig:
    store_nullable_bools_as_objects: bool = False
    store_dates_as_timestamps: bool = False
    store_enum_members_as: Literal["members", "names", "values"] = "members"
    store_nullable_ints_as_floats: bool = False


@dataclasses.dataclass
class GeneralPyaType(abc.ABC):
    field_name: str
    annotated_type: type
    annotated_with: type | None
    nullable: bool
    config: PyaTypesConfig

    def validator(self, df: pd.DataFrame, **kwargs: dict[str, Any]) -> pa.Column:
        self.prevalidate_column(df)
        try:
            return pa.Column(self.annotated_type, nullable=self.nullable, **kwargs)
        except TypeError:
            return pa.Column(object, nullable=self.nullable, **kwargs)

    def prevalidate_column(self, df: pd.DataFrame) -> None:
        return

    def process_getter_value(self, value: Any) -> Any:
        if not hasattr(value, "__len__") and pd.isna(value):
            return None
        elif (
            type(self.annotated_type) is type and type(value) is not self.annotated_type
        ):
            return self.annotated_type(value)
        else:
            return value

    def process_setter_value(self, value: Any) -> Any:
        return value


@dataclasses.dataclass
class BooleanPyaType(GeneralPyaType):
    @override
    def validator(self, df: pd.DataFrame, **kwargs: dict[str, Any]) -> pa.Column:
        self.prevalidate_column(df)
        if not self.nullable:
            return pa.Column(bool, nullable=self.nullable, **kwargs)
        if self.config.store_nullable_bools_as_objects:
            return pa.Column(object, nullable=True, **kwargs)
        else:
            return pa.Column(pd.BooleanDtype(), nullable=True, **kwargs)

    @override
    def prevalidate_column(self, df: pd.DataFrame) -> None:
        if not self.nullable:
            return
        if self.config.store_nullable_bools_as_objects:
            df[self.field_name] = df[self.field_name].astype(object)
        else:
            df[self.field_name] = df[self.field_name].astype(pd.BooleanDtype())
            warnings.warn(
                "Using Pandas' experimental `BooleanDtype` for nullable bool field. "
                "To avoid this, set "
                "`<ObjectsBackingDataframe instance>.store_nullable_bools_as_objects = True`."
            )


@dataclasses.dataclass
class DatePyaType(GeneralPyaType):
    @override
    def validator(self, df: pd.DataFrame, **kwargs: dict[str, Any]) -> pa.Column:
        self.prevalidate_column(df)
        if self.config.store_dates_as_timestamps:
            return pa.Column(DateTime(tz=None), nullable=self.nullable, **kwargs)
        else:
            return pa.Column(object, nullable=self.nullable, **kwargs)

    @override
    def prevalidate_column(self, df: pd.DataFrame) -> None:
        if self.config.store_dates_as_timestamps:
            df[self.field_name] = pd.to_datetime(df[self.field_name])

    @override
    def process_getter_value(self, value: dt.date | None | pd.Timestamp) -> dt.date | None:
        if self.config.store_dates_as_timestamps:
            if value is pd.NaT:
                return None
            else:
                return value.date()
        else:
            return value


@dataclasses.dataclass
class EnumPyaType(GeneralPyaType):
    @property
    def enum_values(self) -> set[Any]:
        return {member.value for member in self.annotated_type}

    @property
    def enum_values_type(self) -> type:
        try:
            return get_exactly_one(set(type(x) for x in self.enum_values))
        except ValueError:
            warnings.warn(
                f"The member values of enum `{self.annotated_type.__name__}` are not all of the "
                "same type. Using `object`."
            )
            return object

    @override
    def validator(self, df: pd.DataFrame, **kwargs: dict[str, Any]) -> pa.Column:
        self.prevalidate_column(df)
        if self.config.store_enum_members_as == "members":
            isin = set(self.annotated_type)
            return pa.Column(
                object,
                nullable=self.nullable,
                checks=[pa.Check(lambda ser: ser.isin(isin))],
            )
        elif self.config.store_enum_members_as == "names":
            isin = set(member.name for member in self.annotated_type)
            return pa.Column(
                str,
                nullable=self.nullable,
                checks=[pa.Check(lambda ser: ser.isin(isin))],
            )
        elif self.config.store_enum_members_as == "values":
            enum_value_pya_type = find_pya_type(
                field_name=self.field_name,
                annotated_type=self.enum_values_type,
                annotated_with=self.annotated_with,
                nullable=self.nullable,
                config=self.config,
            )
            return enum_value_pya_type.validator(
                df, checks=[pa.Check(lambda ser: ser.isin(self.enum_values))]
            )

    @override
    def prevalidate_column(self, df: pd.DataFrame) -> None:
        if self.config.store_enum_members_as == "names":
            df[self.field_name] = (
                df[self.field_name]
                .apply(lambda x: x.name if isinstance(x, self.annotated_type) else x)
                .astype(str)
            )
        elif self.config.store_enum_members_as == "values":
            df[self.field_name] = df[self.field_name].apply(
                lambda x: x.value if isinstance(x, self.annotated_type) else x
            )
            try:
                df[self.field_name] = df[self.field_name].astype(self.enum_values_type)
            except TypeError:
                pass

    @override
    def process_getter_value(self, value: Any) -> Any:
        if self.config.store_enum_members_as == "members":
            return value
        elif self.config.store_enum_members_as == "names":
            return self.annotated_type[value]
        elif self.config.store_enum_members_as == "values":
            return self.annotated_type(value)


@dataclasses.dataclass
class IntegerPyaType(GeneralPyaType):
    @override
    def validator(self, df: pd.DataFrame, **kwargs: dict[str, Any]) -> pa.Column:
        self.prevalidate_column(df)
        if not self.nullable:
            return pa.Column(int, nullable=self.nullable, **kwargs)
        if self.config.store_nullable_ints_as_floats:
            return pa.Column(float, nullable=True, **kwargs)
        else:
            return pa.Column(pd.Int64Dtype(), nullable=True, **kwargs)

    @override
    def prevalidate_column(self, df: pd.DataFrame) -> None:
        if not self.nullable:
            return
        if self.config.store_nullable_ints_as_floats:
            df[self.field_name] = df[self.field_name].astype(float)
        else:
            df[self.field_name] = df[self.field_name].astype(pd.Int64Dtype())
            warnings.warn(
                "Using Pandas' experimental `Int64Dtype` for nullable integer field. "
                "To avoid this, set "
                "`<ObjectsBackingDataframe instance>.store_nullable_ints_as_floats = True`."
            )


@dataclasses.dataclass
class LiteralPyaType(GeneralPyaType):
    @property
    def literal_values(self) -> set[Any]:
        return {value for value in typing.get_args(self.annotated_type)}

    @property
    def literal_values_type(self) -> type:
        try:
            return get_exactly_one(set(type(x) for x in self.literal_values))
        except ValueError:
            return object

    @override
    def validator(self, df: pd.DataFrame, **kwargs) -> pa.Column:
        self.prevalidate_column(df)
        enum_value_pya_type = find_pya_type(
            field_name=self.field_name,
            annotated_type=self.literal_values_type,
            annotated_with=self.annotated_with,
            nullable=self.nullable,
            config=self.config,
        )
        return enum_value_pya_type.validator(
            df, checks=[pa.Check(lambda ser: ser.isin(self.literal_values))]
        )

    @override
    def prevalidate_column(self, df: pd.DataFrame) -> None:
        try:
            df[self.field_name] = df[self.field_name].astype(self.literal_values_type)
        except (TypeError, ValueError):
            pass


@dataclasses.dataclass
class TimedeltaPyaType(GeneralPyaType):
    @override
    def process_getter_value(self, value: pd.Timedelta) -> pd.Timedelta | dt.timedelta | None:
        if value is pd.NaT:
            return None
        elif issubclass(self.annotated_type, pd.Timedelta):
            return value
        else:
            return value.to_pytimedelta()

@dataclasses.dataclass
class DatetimeyPyaType(GeneralPyaType):
    @override
    def validator(self, df: pd.DataFrame, **kwargs) -> pa.Column:
        self.prevalidate_column(df)
        return pa.Column(self.annotated_with, nullable=self.nullable, **kwargs)

    @override
    def process_getter_value(self, value: pd.Timestamp) -> pd.Timestamp | dt.datetime | None:
        if value is pd.NaT:
            return None
        elif issubclass(self.annotated_type, pd.Timestamp):
            return value
        else:
            return value.to_pydatetime()


@dataclasses.dataclass
class StringPyaType(GeneralPyaType):
    @override
    def validator(self, df: pd.DataFrame, **kwargs) -> pa.Column:
        self.prevalidate_column(df)
        return pa.Column(object, nullable=self.nullable, **kwargs)


def find_pya_type(
    field_name: str,
    annotated_type: type,
    annotated_with: type | None,
    nullable: bool,
    config: PyaTypesConfig,
) -> GeneralPyaType:
    args = (field_name, annotated_type, annotated_with, nullable, config)
    if annotated_type is bool:
        return BooleanPyaType(*args)
    elif annotated_type is dt.date:
        return DatePyaType(*args)
    elif isinstance(annotated_type, EnumType):
        return EnumPyaType(*args)
    elif annotated_type is int:
        return IntegerPyaType(*args)
    elif isinstance(annotated_type, typing._LiteralGenericAlias):
        return LiteralPyaType(*args)
    elif annotated_type is pd.Timedelta or annotated_type is dt.timedelta:
        return TimedeltaPyaType(*args)
    elif annotated_type is pd.Timestamp or annotated_type is dt.datetime:
        return DatetimeyPyaType(*args)
    elif annotated_type is str:
        return StringPyaType(*args)
    else:
        return GeneralPyaType(*args)
