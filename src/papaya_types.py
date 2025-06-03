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
class PapayaTypesConfig:
    store_nullable_bools_as_objects: bool = False
    store_dates_as_timestamps: bool = False
    store_enum_members_as: Literal["members", "names", "values"] = "members"
    store_nullable_ints_as_floats: bool = False


@dataclasses.dataclass
class GeneralPapayaType(abc.ABC):
    field_name: str
    annotated_type: type
    annotated_with: type | None
    nullable: bool
    config: PapayaTypesConfig

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
        if type(self.annotated_type) is type and type(value) is not self.annotated_type:
            if not self.nullable:
                raise TypeError(f"{value} must be {self.annotated_type}.")
            elif value is not None:
                raise TypeError(f"{value} must be {self.annotated_type} or None.")
        return value


@dataclasses.dataclass
class BooleanPapayaType(GeneralPapayaType):
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
                "`<ObjectsBackingDataframe instance>.store_nullable_bools_as_objects "
                "= True`.",
                stacklevel=2,
            )


@dataclasses.dataclass
class DatePapayaType(GeneralPapayaType):
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
    def process_getter_value(
        self, value: dt.date | None | pd.Timestamp
    ) -> dt.date | None:
        if self.config.store_dates_as_timestamps:
            if value is pd.NaT:
                return None
            else:
                return value.date()
        else:
            return value


@dataclasses.dataclass
class EnumPapayaType(GeneralPapayaType):
    @property
    def enum_members(self) -> set[Any]:
        return set(member for member in self.annotated_type)

    @property
    def enum_names(self) -> set[Any]:
        return set(member.name for member in self.annotated_type)

    @property
    def enum_values(self) -> set[Any]:
        return set(member.value for member in self.annotated_type)

    @property
    def enum_values_type(self) -> type:
        try:
            return get_exactly_one(set(type(x) for x in self.enum_values))
        except ValueError:
            warnings.warn(
                f"The member values of enum `{self.annotated_type.__name__}` are not "
                "all of the same type. Using `object`.",
                stacklevel=2,
            )
            return object

    @override
    def validator(self, df: pd.DataFrame, **kwargs: dict[str, Any]) -> pa.Column:
        self.prevalidate_column(df)
        if self.config.store_enum_members_as == "members":
            return pa.Column(
                object,
                nullable=self.nullable,
                checks=[pa.Check(lambda ser: ser.isin(self.enum_members))],
            )
        elif self.config.store_enum_members_as == "names":
            return pa.Column(
                str,
                nullable=self.nullable,
                checks=[pa.Check(lambda ser: ser.isin(self.enum_names))],
            )
        elif self.config.store_enum_members_as == "values":
            enum_value_papaya_type = find_papaya_type(
                field_name=self.field_name,
                annotated_type=self.enum_values_type,
                annotated_with=self.annotated_with,
                nullable=self.nullable,
                config=self.config,
            )
            return enum_value_papaya_type.validator(
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

    def process_setter_value(self, value: Any) -> Any:
        if value not in self.enum_members:
            if not self.nullable:
                raise ValueError(f"{value} must be in {self.annotated_type.__name__}.")
            elif value is not None:
                raise ValueError(
                    f"{value} must be in {self.annotated_type.__name__} or None."
                )

        if self.config.store_enum_members_as == "members":
            return value
        elif self.config.store_enum_members_as == "names":
            return value.name
        elif self.config.store_enum_members_as == "values":
            return value.value


@dataclasses.dataclass
class IntegerPapayaType(GeneralPapayaType):
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
                "`<ObjectsBackingDataframe instance>.store_nullable_ints_as_floats "
                "= True`.",
                stacklevel=2,
            )


@dataclasses.dataclass
class LiteralPapayaType(GeneralPapayaType):
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
        enum_value_papaya_type = find_papaya_type(
            field_name=self.field_name,
            annotated_type=self.literal_values_type,
            annotated_with=self.annotated_with,
            nullable=self.nullable,
            config=self.config,
        )
        return enum_value_papaya_type.validator(
            df, checks=[pa.Check(lambda ser: ser.isin(self.literal_values))]
        )

    @override
    def prevalidate_column(self, df: pd.DataFrame) -> None:
        try:
            df[self.field_name] = df[self.field_name].astype(self.literal_values_type)
        except (TypeError, ValueError):
            pass

    @override
    def process_getter_value(self, value: Any) -> Any:
        if not hasattr(value, "__len__") and pd.isna(value):
            return None
        elif (
            type(self.literal_values_type) is type
            and type(value) is not self.literal_values_type
            and self.literal_values_type is not object
        ):
            return self.literal_values_type(value)
        else:
            return value

    def process_setter_value(self, value: Any) -> Any:
        if value not in self.literal_values:
            if not self.nullable:
                raise ValueError(f"{value} must be in {self.literal_values}.")
            elif value is not None:
                raise ValueError(f"{value} must be in {self.literal_values} or None.")

        return value


@dataclasses.dataclass
class TimedeltaPapayaType(GeneralPapayaType):
    @override
    def process_getter_value(
        self, value: pd.Timedelta
    ) -> pd.Timedelta | dt.timedelta | None:
        if value is pd.NaT:
            return None
        elif issubclass(self.annotated_type, pd.Timedelta):
            return value
        else:
            return value.to_pytimedelta()


@dataclasses.dataclass
class DatetimeyPapayaType(GeneralPapayaType):
    @override
    def validator(self, df: pd.DataFrame, **kwargs) -> pa.Column:
        self.prevalidate_column(df)
        return pa.Column(self.annotated_with, nullable=self.nullable, **kwargs)

    @override
    def process_getter_value(
        self, value: pd.Timestamp
    ) -> pd.Timestamp | dt.datetime | None:
        if value is pd.NaT:
            return None
        elif issubclass(self.annotated_type, pd.Timestamp):
            return value
        else:
            return value.to_pydatetime()


@dataclasses.dataclass
class StringPapayaType(GeneralPapayaType):
    @override
    def validator(self, df: pd.DataFrame, **kwargs) -> pa.Column:
        self.prevalidate_column(df)
        return pa.Column(object, nullable=self.nullable, **kwargs)


def find_papaya_type(
    field_name: str,
    annotated_type: type,
    annotated_with: type | None,
    nullable: bool,
    config: PapayaTypesConfig,
) -> GeneralPapayaType:
    args = (field_name, annotated_type, annotated_with, nullable, config)
    if annotated_type is bool:
        return BooleanPapayaType(*args)
    elif annotated_type is dt.date:
        return DatePapayaType(*args)
    elif isinstance(annotated_type, EnumType):
        return EnumPapayaType(*args)
    elif annotated_type is int:
        return IntegerPapayaType(*args)
    elif isinstance(annotated_type, typing._LiteralGenericAlias):
        return LiteralPapayaType(*args)
    elif annotated_type is pd.Timedelta or annotated_type is dt.timedelta:
        return TimedeltaPapayaType(*args)
    elif annotated_type is pd.Timestamp or annotated_type is dt.datetime:
        return DatetimeyPapayaType(*args)
    elif annotated_type is str:
        return StringPapayaType(*args)
    else:
        return GeneralPapayaType(*args)
