import dataclasses
import datetime as dt
from enum import Enum
from typing import Annotated, Literal, Optional, Union
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pandera.engines.pandas_engine import DateTime

from objects_backing_dataframe import (
    ObjectsBackingDataframe,
    SettingOnIndexLevelError,
    dataframe_backed_object,
)
from objects_dataframe_base import DataframeIndex
from papaya_config import PapayaConfig


def test_dataframe_backed() -> None:
    @dataframe_backed_object
    @dataclasses.dataclass
    class User:
        id: int
        name: str

    UserDataframe = ObjectsBackingDataframe[User]  # noqa: N806

    user_df = UserDataframe(
        [
            User(id=1, name="a"),
            User(id=2, name="b"),
        ]
    )

    # The `UserDataFrame` instance can be iterated over yielding `User` instances:
    for user in user_df:
        assert isinstance(user, User)
    df_iterator = iter(user_df)
    user_1 = next(df_iterator)
    user_2 = next(df_iterator)
    assert user_1 == User(id=1, name="a")
    assert user_2 == User(id=2, name="b")
    # Values stored as, for example, `np.int64` in the backing dataframe are cast to
    #     `int` when accessed as instance attributes:
    assert isinstance(user_1.id, int)

    # Updating an attribute updates the corresponding value in the `UserDataFrame`
    #     instance:
    user_1.name = "c"
    assert user_df.loc[0, "name"] == "c"

    # The `User` instances reference a shared `UserDataFrame` instance:
    assert isinstance(user_1.__df, pd.DataFrame)
    assert user_1.__df is user_2.__df
    assert user_1.__df is user_df

    # Even if the variable `user_df` is deleted, the `UserDataFrame` instance is still
    #     in memory and accessible by `User` instances:
    del user_df
    assert user_1.name == "c"
    assert isinstance(user_1.__df, pd.DataFrame)
    assert user_1.__df is user_2.__df


class AnEnum(Enum):
    A = 1
    B = 2
    C = 3


class TestCompatibilityWithNonNullableDataTypes:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        bool_field: bool
        date_field: dt.date
        enum_field: AnEnum
        float_field: float
        int_field: int
        list_field: list[int]
        literal_field: Literal[1, 2, 3]
        pandas_timedelta_field: pd.Timedelta
        pandas_timestamp_w_tzinfo_field: Annotated[
            pd.Timestamp, DateTime(tz=ZoneInfo("America/Toronto"))
        ]
        pandas_timestamp_wo_tzinfo_field: Annotated[pd.Timestamp, DateTime(tz=None)]
        pydatetime_w_tzinfo_field: Annotated[
            dt.datetime, DateTime(tz=ZoneInfo("America/Toronto"))
        ]
        pydatetime_wo_tzinfo_field: Annotated[dt.datetime, DateTime(tz=None)]
        pytimedelta_field: dt.timedelta
        str_field: str
        time_w_tzinfo_field: dt.time
        time_wo_tzinfo_field: dt.time
        uuid_field: UUID
        zone_info_field: ZoneInfo

    foo_instance = Foo(
        bool_field=True,
        date_field=dt.date(2000, 4, 2),
        enum_field=AnEnum.A,
        float_field=4.2,
        int_field=42,
        list_field=[4, 2],
        literal_field=1,
        pandas_timedelta_field=pd.Timedelta(days=4, hours=2),
        pandas_timestamp_w_tzinfo_field=pd.Timestamp(
            "2000-04-02 23:59", tz="America/Toronto"
        ),
        pandas_timestamp_wo_tzinfo_field=pd.Timestamp("2000-04-02 23:59"),
        pydatetime_w_tzinfo_field=dt.datetime(
            2000, 4, 2, 23, 59, tzinfo=ZoneInfo("America/Toronto")
        ),
        pydatetime_wo_tzinfo_field=dt.datetime(2000, 4, 2),  # noqa: DTZ001
        pytimedelta_field=dt.timedelta(days=4, hours=2),
        str_field="bar",
        time_w_tzinfo_field=dt.time(23, 59, tzinfo=ZoneInfo("America/Toronto")),
        time_wo_tzinfo_field=dt.time(23, 59),
        uuid_field=uuid4(),
        zone_info_field=ZoneInfo("America/Toronto"),
    )

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_with_no_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.foo_instance,
            ]
        )
        self._check_dtypes(foo_df)
        (foo_0,) = list(foo_df)
        _check_foo_types(foo_0)

        foo_0 = _change_foo_values(foo_0)
        self._check_dtypes(foo_df)
        (foo_0,) = list(foo_df)
        _check_foo_types(foo_0)
        _check_changed_foo_values(foo_0)

    @staticmethod
    def _check_dtypes(df: ObjectsBackingDataframe) -> None:
        assert len(df.dtypes) == 18
        assert df.dtypes["bool_field"] == np.dtype("bool")
        assert df.dtypes["date_field"] == np.dtype("O")
        assert df.dtypes["enum_field"] == np.dtype("O")
        assert df.dtypes["float_field"] == np.dtype("float64")
        assert df.dtypes["int_field"] == np.dtype("int64")
        assert df.dtypes["list_field"] == np.dtype("O")
        assert df.dtypes["literal_field"] == np.dtype("int64")
        assert df.dtypes["pandas_timedelta_field"] == np.dtype("<m8[ns]")
        assert df.dtypes["pandas_timestamp_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pandas_timestamp_wo_tzinfo_field"] == np.dtype("<M8[ns]")
        assert df.dtypes["pydatetime_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pydatetime_wo_tzinfo_field"] == np.dtype("<M8[ns]")
        assert df.dtypes["pytimedelta_field"] == np.dtype("<m8[ns]")
        assert df.dtypes["str_field"] == np.dtype("O")
        assert df.dtypes["time_w_tzinfo_field"] == np.dtype("O")
        assert df.dtypes["time_wo_tzinfo_field"] == np.dtype("O")
        assert df.dtypes["uuid_field"] == np.dtype("O")
        assert df.dtypes["zone_info_field"] == np.dtype("O")


class TestCompatibilityWithNullableDataTypes:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        bool_field: bool | None = None
        date_field: dt.date | None = None
        enum_field: AnEnum | None = None
        float_field: float | None = None
        int_field: int | None = None
        list_field: list[int] | None = None
        literal_field: Literal[1, 2, 3, None] = None
        pandas_timedelta_field: pd.Timedelta | None = None
        pandas_timestamp_w_tzinfo_field: (
            Annotated[pd.Timestamp, DateTime(tz=ZoneInfo("America/Toronto"))] | None
        ) = None
        pandas_timestamp_wo_tzinfo_field: (
            Annotated[pd.Timestamp, DateTime(tz=None)] | None
        ) = None
        pydatetime_w_tzinfo_field: (
            Annotated[dt.datetime, DateTime(tz=ZoneInfo("America/Toronto"))] | None
        ) = None
        pydatetime_wo_tzinfo_field: Annotated[dt.datetime, DateTime(tz=None)] | None = (
            None
        )
        pytimedelta_field: dt.timedelta | None = None
        str_field: str | None = None
        time_w_tzinfo_field: dt.time | None = None
        time_wo_tzinfo_field: dt.time | None = None
        uuid_field: UUID | None = None
        zone_info_field: ZoneInfo | None = None

    foo_instance = Foo(
        bool_field=True,
        date_field=dt.date(2000, 4, 2),
        enum_field=AnEnum.A,
        float_field=4.2,
        int_field=42,
        list_field=[4, 2],
        literal_field=1,
        pandas_timedelta_field=pd.Timedelta(days=4, hours=2),
        pandas_timestamp_w_tzinfo_field=pd.Timestamp(
            "2000-04-02 23:59", tz="America/Toronto"
        ),
        pandas_timestamp_wo_tzinfo_field=pd.Timestamp("2000-04-02 23:59"),
        pydatetime_w_tzinfo_field=dt.datetime(
            2000, 4, 2, 23, 59, tzinfo=ZoneInfo("America/Toronto")
        ),
        pydatetime_wo_tzinfo_field=dt.datetime(2000, 4, 2),  # noqa: DTZ001
        pytimedelta_field=dt.timedelta(days=4, hours=2),
        str_field="bar",
        time_w_tzinfo_field=dt.time(23, 59, tzinfo=ZoneInfo("America/Toronto")),
        time_wo_tzinfo_field=dt.time(23, 59),
        uuid_field=uuid4(),
        zone_info_field=ZoneInfo("America/Toronto"),
    )

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_with_no_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.foo_instance,
            ]
        )
        self._check_dtypes(foo_df)
        (foo_0,) = list(foo_df)
        _check_foo_types(foo_0)

        foo_0 = _change_foo_values(foo_0)
        self._check_dtypes(foo_df)
        (foo_0,) = list(foo_df)
        _check_foo_types(foo_0)
        _check_changed_foo_values(foo_0)

    def test_with_some_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.foo_instance,
                self.Foo(),  # All `None`s.
            ]
        )
        self._check_dtypes(foo_df)
        foo_0, foo_1 = list(foo_df)
        _check_foo_types(foo_0)
        for field in dataclasses.fields(self.Foo):
            assert getattr(foo_1, field.name) is None

        foo_0 = _change_foo_values(foo_0)
        self._check_dtypes(foo_df)
        foo_0, foo_1 = list(foo_df)
        _check_foo_types(foo_0)
        _check_changed_foo_values(foo_0)

    def test_with_all_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(),  # All `None`s.
            ]
        )
        self._check_dtypes(foo_df)
        (foo_0,) = list(foo_df)
        for field in dataclasses.fields(self.Foo):
            assert getattr(foo_0, field.name) is None

        foo_0 = _change_foo_values(foo_0)
        self._check_dtypes(foo_df)
        (foo_0,) = list(foo_df)
        _check_foo_types(foo_0)
        _check_changed_foo_values(foo_0)

    @staticmethod
    def _check_dtypes(df: ObjectsBackingDataframe) -> None:
        assert len(df.dtypes) == 18
        assert df.dtypes["bool_field"] == pd.BooleanDtype()
        assert df.dtypes["date_field"] == np.dtype("O")
        assert df.dtypes["float_field"] == np.dtype("float64")
        assert df.dtypes["int_field"] == pd.Int64Dtype()
        assert df.dtypes["list_field"] == np.dtype("O")
        assert df.dtypes["literal_field"] == pd.Int64Dtype()
        assert df.dtypes["pandas_timedelta_field"] == np.dtype("<m8[ns]")
        assert df.dtypes["pandas_timestamp_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pandas_timestamp_wo_tzinfo_field"] == np.dtype("<M8[ns]")
        assert df.dtypes["pydatetime_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pydatetime_wo_tzinfo_field"] == np.dtype("<M8[ns]")
        assert df.dtypes["pytimedelta_field"] == np.dtype("<m8[ns]")
        assert df.dtypes["str_field"] == np.dtype("O")
        assert df.dtypes["time_w_tzinfo_field"] == np.dtype("O")
        assert df.dtypes["time_wo_tzinfo_field"] == np.dtype("O")
        assert df.dtypes["uuid_field"] == np.dtype("O")
        assert df.dtypes["zone_info_field"] == np.dtype("O")


def test_storing_dates_as_timestamps():
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        date_field: dt.date

        papaya_config = PapayaConfig(store_dates_as_timestamps=True)

    FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
    foo_df = FooDataframe([Foo(date_field=dt.date(2000, 4, 2))])
    assert foo_df.dtypes["date_field"] == np.dtype("<M8[ns]")
    assert type(foo_df.loc[0, "date_field"]) is pd.Timestamp
    assert foo_df.loc[0, "date_field"] == pd.Timestamp("2000-04-02")
    (foo_0,) = list(foo_df)
    assert type(foo_0.date_field) is dt.date
    assert foo_0.date_field == dt.date(2000, 4, 2)

    foo_df.loc[0, "date_field"] = pd.Timestamp("2000-04-02", tz="America/Toronto")
    with pytest.raises(pa.errors.SchemaError):
        foo_df.validate()


class TestCompatibilityWithEnums:

    def test_storing_enum_members_as_members(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            enum_field: AnEnum

            papaya_config = PapayaConfig(store_enum_members_as="members")

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
        foo_df = FooDataframe([Foo(enum_field=AnEnum.A)])
        assert foo_df.dtypes["enum_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "enum_field"]) is AnEnum
        assert foo_df.loc[0, "enum_field"] is AnEnum.A
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        foo_0.enum_field = AnEnum.B
        (foo_0,) = list(foo_df)
        assert foo_0.enum_field is AnEnum.B

        with pytest.raises(pa.errors.SchemaError):
            FooDataframe([Foo(enum_field="<invalid-value>")])

    def test_storing_enum_members_as_names(self) -> None:

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            enum_field: AnEnum

            papaya_config = PapayaConfig(store_enum_members_as="names")

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
        foo_df = FooDataframe([Foo(enum_field=AnEnum.A)])
        assert foo_df.dtypes["enum_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "enum_field"]) is str
        assert foo_df.loc[0, "enum_field"] == "A"
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        foo_0.enum_field = AnEnum.B
        (foo_0,) = list(foo_df)
        assert foo_0.enum_field is AnEnum.B

        with pytest.raises(pa.errors.SchemaError):
            FooDataframe([Foo(enum_field="<invalid-value>")])

    def test_storing_enum_members_as_integer_values(self) -> None:

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            enum_field: AnEnum

            papaya_config = PapayaConfig(store_enum_members_as="values")

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
        foo_df = FooDataframe([Foo(enum_field=AnEnum.A)])
        assert foo_df.dtypes["enum_field"] == np.dtype("int64")
        assert type(foo_df.loc[0, "enum_field"]) is np.int64
        assert foo_df.loc[0, "enum_field"] == 1
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        foo_0.enum_field = AnEnum.B
        (foo_0,) = list(foo_df)
        assert foo_0.enum_field is AnEnum.B

        with pytest.raises(pa.errors.SchemaError):
            FooDataframe([Foo(enum_field="<invalid-value>")])

    def test_storing_enum_members_as_string_values(self) -> None:
        class AnEnum(Enum):
            A = "a"
            B = "b"
            C = "c"

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            enum_field: AnEnum

            papaya_config = PapayaConfig(store_enum_members_as="values")

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
        foo_df = FooDataframe([Foo(enum_field=AnEnum.A)])
        assert foo_df.dtypes["enum_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "enum_field"]) is str
        assert foo_df.loc[0, "enum_field"] == "a"
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        with pytest.raises(pa.errors.SchemaError):
            FooDataframe([Foo(enum_field="<invalid-value>")])

    def test_storing_enum_members_as_object_values(self) -> None:
        class AnEnum(Enum):
            A = ZoneInfo("America/Toronto")
            B = ZoneInfo("Asia/Kolkata")
            C = ZoneInfo("Asia/Tokyo")

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            enum_field: AnEnum

            papaya_config = PapayaConfig(store_enum_members_as="values")

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
        foo_df = FooDataframe([Foo(enum_field=AnEnum.A)])
        assert foo_df.dtypes["enum_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "enum_field"]) is ZoneInfo
        assert foo_df.loc[0, "enum_field"] == ZoneInfo("America/Toronto")
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A


class TestCompatibilityWithLiterals:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        literal_field: Literal[1, True, "a", AnEnum.A]

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_with_mixed_type_values(self) -> None:
        foo_df = self.FooDataframe([self.Foo(literal_field=1)])
        assert foo_df.dtypes["literal_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "literal_field"]) is int
        assert foo_df.loc[0, "literal_field"] == 1
        (foo_0,) = list(foo_df)
        assert type(foo_0.literal_field) is int
        assert foo_0.literal_field == 1

    def test_validating_literal_values(self) -> None:
        with pytest.raises(pa.errors.SchemaError):
            self.FooDataframe([self.Foo(literal_field="<invalid-value>")])


def _check_foo_types(foo_instance: type) -> None:
    assert type(foo_instance.bool_field) is bool
    assert type(foo_instance.date_field) is dt.date
    assert type(foo_instance.enum_field) is AnEnum
    assert type(foo_instance.float_field) is float
    assert type(foo_instance.int_field) is int
    assert type(foo_instance.list_field) is list
    assert type(foo_instance.literal_field) is int
    assert type(foo_instance.pandas_timedelta_field) is pd.Timedelta
    assert type(foo_instance.pandas_timestamp_w_tzinfo_field) is pd.Timestamp
    assert type(foo_instance.pandas_timestamp_wo_tzinfo_field) is pd.Timestamp
    assert type(foo_instance.pydatetime_w_tzinfo_field) is dt.datetime
    assert type(foo_instance.pydatetime_wo_tzinfo_field) is dt.datetime
    assert type(foo_instance.pytimedelta_field) is dt.timedelta
    assert type(foo_instance.str_field) is str
    assert type(foo_instance.time_w_tzinfo_field) is dt.time
    assert type(foo_instance.time_wo_tzinfo_field) is dt.time
    assert type(foo_instance.uuid_field) is UUID
    assert type(foo_instance.zone_info_field) is ZoneInfo


def _change_foo_values(foo_instance: type) -> type:
    foo_instance.bool_field = False
    foo_instance.date_field = dt.date(2000, 7, 3)
    foo_instance.enum_field = AnEnum.B
    foo_instance.float_field = 7.3
    foo_instance.int_field = 73
    foo_instance.list_field = [7, 3]
    foo_instance.literal_field = 2
    foo_instance.pandas_timedelta_field = pd.Timedelta(days=7, hours=3)
    foo_instance.pandas_timestamp_w_tzinfo_field = pd.Timestamp(
        "2000-07-03 00:00", tz="America/Toronto"
    )
    foo_instance.pandas_timestamp_wo_tzinfo_field = pd.Timestamp("2000-07-03 00:00")
    foo_instance.pydatetime_w_tzinfo_field = dt.datetime(
        2000, 7, 3, 0, 0, tzinfo=ZoneInfo("America/Toronto")
    )
    foo_instance.pydatetime_wo_tzinfo_field = dt.datetime(2000, 7, 3)  # noqa: DTZ001
    foo_instance.pytimedelta_field = dt.timedelta(days=7, hours=3)
    foo_instance.str_field = "baz"
    foo_instance.time_w_tzinfo_field = dt.time(0, 0, tzinfo=ZoneInfo("America/Toronto"))
    foo_instance.time_wo_tzinfo_field = dt.time(0, 0)
    foo_instance.uuid_field = uuid4()
    foo_instance.zone_info_field = ZoneInfo("Asia/Kolkata")

    return foo_instance


def _check_changed_foo_values(foo_instance: type) -> None:
    assert foo_instance.bool_field is False
    assert foo_instance.date_field == dt.date(2000, 7, 3)
    assert foo_instance.enum_field is AnEnum.B
    assert foo_instance.float_field == 7.3
    assert foo_instance.int_field == 73
    assert foo_instance.list_field == [7, 3]
    assert foo_instance.literal_field == 2
    assert foo_instance.pandas_timedelta_field == pd.Timedelta(days=7, hours=3)
    assert foo_instance.pandas_timestamp_w_tzinfo_field == pd.Timestamp(
        "2000-07-03 00:00", tz="America/Toronto"
    )
    assert foo_instance.pandas_timestamp_wo_tzinfo_field == pd.Timestamp(
        "2000-07-03 00:00"
    )
    assert foo_instance.pydatetime_w_tzinfo_field == dt.datetime(
        2000, 7, 3, 0, 0, tzinfo=ZoneInfo("America/Toronto")
    )
    assert foo_instance.pydatetime_wo_tzinfo_field == dt.datetime(  # noqa: DTZ001
        2000, 7, 3
    )
    assert foo_instance.pytimedelta_field == dt.timedelta(days=7, hours=3)
    assert foo_instance.str_field == "baz"
    assert foo_instance.time_w_tzinfo_field == dt.time(
        0, 0, tzinfo=ZoneInfo("America/Toronto")
    )
    assert foo_instance.time_wo_tzinfo_field == dt.time(0, 0)
    assert foo_instance.zone_info_field == ZoneInfo("Asia/Kolkata")


def test_raising_for_prohibited_datetimey_type_annotations() -> None:
    with pytest.raises(TypeError):

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            pandas_timestamp_field: pd.Timestamp

    with pytest.raises(TypeError):

        @dataframe_backed_object
        @dataclasses.dataclass
        class Bar:
            pydatetime_field: dt.datetime


class TestCompatibilityWithOddTypeAnnotations:
    def test_compatibility_with_different_nullable_type_annotations(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            union_nullable_field_1: str | None = None
            union_nullable_field_2: Union[str, None] = None
            optional_field: Optional[str] = None

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
        foo_df = FooDataframe(
            [
                Foo(),  # All `None`s.
                Foo(
                    union_nullable_field_1="a",
                    union_nullable_field_2="b",
                    optional_field="c",
                ),
            ]
        )
        foo_0, foo_1 = list(foo_df)
        assert foo_0.union_nullable_field_1 is None
        assert foo_0.union_nullable_field_2 is None
        assert foo_0.optional_field is None
        assert foo_1.union_nullable_field_1 == "a"
        assert foo_1.union_nullable_field_2 == "b"
        assert foo_1.optional_field == "c"

    def test_compatibility_with_single_union_type(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            single_union_field: Union[int]
            # ^ `Field.type` is actually just `int`, not `Union[int]`, so this is never
            #     a problem.

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806
        foo_df = FooDataframe([Foo(single_union_field=1)])
        (foo_0,) = list(foo_df)
        assert foo_0.single_union_field == 1


class TestStoringNullableIntsAsFloats:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        int_field: int
        nullable_int_field: int | None

        papaya_config = PapayaConfig(store_nullable_ints_as_floats=True)

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_with_no_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(int_field=1, nullable_int_field=2),
            ]
        )
        assert foo_df.dtypes["int_field"] == np.dtype("int64")
        assert foo_df.dtypes["nullable_int_field"] == np.dtype("float64")

    def test_with_some_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(int_field=1, nullable_int_field=None),
                self.Foo(int_field=1, nullable_int_field=2),
            ]
        )
        assert foo_df.dtypes["int_field"] == np.dtype("int64")
        assert foo_df.dtypes["nullable_int_field"] == np.dtype("float64")

    def test_with_all_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(int_field=1, nullable_int_field=None),
            ]
        )
        assert foo_df.dtypes["int_field"] == np.dtype("int64")
        assert foo_df.dtypes["nullable_int_field"] == np.dtype("float64")


class TestStoringNullableBoolsAsObjects:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        bool_field: bool
        nullable_bool_field: bool | None

        papaya_config = PapayaConfig(store_nullable_bools_as_objects=True)

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_with_no_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(bool_field=True, nullable_bool_field=True),
            ]
        )
        assert foo_df.dtypes["bool_field"] == np.dtype("bool")
        assert foo_df.dtypes["nullable_bool_field"] == np.dtype("O")

    def test_with_some_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(bool_field=True, nullable_bool_field=None),
                self.Foo(bool_field=True, nullable_bool_field=True),
            ]
        )
        assert foo_df.dtypes["bool_field"] == np.dtype("bool")
        assert foo_df.dtypes["nullable_bool_field"] == np.dtype("O")

    def test_with_all_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(bool_field=True, nullable_bool_field=None),
            ]
        )
        assert foo_df.dtypes["bool_field"] == np.dtype("bool")
        assert foo_df.dtypes["nullable_bool_field"] == np.dtype("O")


class TestWithIndex:
    def test_with_set_index_set_to_true(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field: Annotated[int, DataframeIndex]
            non_index_field: float

            papaya_config = PapayaConfig(set_index=True)

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe([Foo(index_field=1, non_index_field=4.2)])
        assert foo_df.index.dtype == np.dtype("int64")
        assert foo_df.dtypes["non_index_field"] == np.dtype("float64")

        with pytest.raises(ValueError):
            FooDataframe(
                pd.DataFrame([Foo(index_field=1, non_index_field=4.2)]).set_index(
                    "index_field"
                )
            )
        with pytest.raises(ValueError):
            FooDataframe(
                pd.DataFrame([Foo(index_field=1, non_index_field=4.2)]).set_index(
                    "non_index_field"
                )
            )

    def test_with_set_index_set_to_false(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field: Annotated[int, DataframeIndex]
            non_index_field: float

            papaya_config = PapayaConfig(set_index=False)

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe(
            pd.DataFrame([Foo(index_field=1, non_index_field=4.2)]).set_index(
                "index_field"
            )
        )
        assert foo_df.index.dtype == np.dtype("int64")
        assert foo_df.dtypes["non_index_field"] == np.dtype("float64")

        with pytest.raises(ValueError):
            FooDataframe([Foo(index_field=1, non_index_field=4.2)])

        with pytest.raises(ValueError):
            FooDataframe(
                pd.DataFrame([Foo(index_field=1, non_index_field=4.2)]).set_index(
                    "index_field", append=True
                )
            )

    def test_ensuring_unique_index(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field: Annotated[int, DataframeIndex]
            non_index_field: float

            papaya_config = PapayaConfig(set_index=True)

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        with pytest.raises(pa.errors.SchemaError):
            FooDataframe(
                [
                    Foo(index_field=1, non_index_field=4.2),
                    Foo(index_field=1, non_index_field=7.3),
                ]
            )

    def test_getting_and_setting(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field: Annotated[int, DataframeIndex]
            non_index_field: float

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe(
            pd.DataFrame([Foo(index_field=1, non_index_field=4.2)]).set_index(
                "index_field"
            )
        )
        (foo_0,) = list(foo_df)

        assert type(foo_0.index_field) is int
        assert foo_0.index_field == 1
        with pytest.raises(SettingOnIndexLevelError):
            foo_0.index_field = 2

        assert foo_0.non_index_field == 4.2
        foo_0.non_index_field = 7.3
        assert foo_0.non_index_field == 7.3

    def test_with_datetimey_index(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field: Annotated[
                Annotated[pd.Timestamp, DateTime(tz=ZoneInfo("America/Toronto"))],
                DataframeIndex,
            ]
            non_index_field: float

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe(
            pd.DataFrame(
                [
                    Foo(
                        index_field=pd.Timestamp(
                            "2000-04-02 23:59", tz="America/Toronto"
                        ),
                        non_index_field=4.2,
                    )
                ]
            ).set_index("index_field")
        )
        assert foo_df.index.dtype == pd.DatetimeTZDtype(tz=ZoneInfo("America/Toronto"))


class TestWithMultiIndex:
    def test_with_set_index_set_to_true(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field_1: Annotated[int, DataframeIndex]
            index_field_2: Annotated[str, DataframeIndex]
            non_index_field: float

            papaya_config = PapayaConfig(set_index=True)

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe(
            [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
        )
        assert foo_df.index.dtypes["index_field_1"] == np.dtype("int64")
        assert foo_df.index.dtypes["index_field_2"] == np.dtype("O")
        assert foo_df.dtypes["non_index_field"] == np.dtype("float64")

        with pytest.raises(ValueError):
            FooDataframe(
                pd.DataFrame(
                    [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
                ).set_index("index_field_1")
            )
        with pytest.raises(ValueError):
            FooDataframe(
                pd.DataFrame(
                    [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
                ).set_index(["index_field_1", "index_field_2"])
            )
        with pytest.raises(ValueError):
            FooDataframe(
                pd.DataFrame(
                    [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
                ).set_index("non_index_field")
            )

    def test_with_set_index_set_to_false(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field_1: Annotated[int, DataframeIndex]
            index_field_2: Annotated[str, DataframeIndex]
            non_index_field: float

            papaya_config = PapayaConfig(set_index=False)

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe(
            pd.DataFrame(
                [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
            ).set_index(["index_field_1", "index_field_2"])
        )
        assert foo_df.index.dtypes["index_field_1"] == np.dtype("int64")
        assert foo_df.index.dtypes["index_field_2"] == np.dtype("O")
        assert foo_df.dtypes["non_index_field"] == np.dtype("float64")

        with pytest.raises(ValueError):
            FooDataframe(
                [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
            )

        with pytest.raises(ValueError):
            FooDataframe(
                pd.DataFrame(
                    [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
                ).set_index(["index_field_1", "index_field_2"], append=True)
            )

    def test_with_datetimey_index(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field_1: Annotated[
                Annotated[pd.Timestamp, DateTime(tz=ZoneInfo("America/Toronto"))],
                DataframeIndex,
            ]
            index_field_2: Annotated[str, DataframeIndex]
            non_index_field: float

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe(
            pd.DataFrame(
                [
                    Foo(
                        index_field_1=pd.Timestamp(
                            "2000-04-02 23:59", tz="America/Toronto"
                        ),
                        index_field_2="bar",
                        non_index_field=4.2,
                    )
                ]
            ).set_index(["index_field_1", "index_field_2"])
        )
        assert foo_df.index.dtypes["index_field_1"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )

    def test_ensuring_unique_multiindex(self) -> None:

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field_1: Annotated[int, DataframeIndex]
            index_field_2: Annotated[str, DataframeIndex]
            non_index_field: float

            papaya_config = PapayaConfig(set_index=True)

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        FooDataframe(
            [
                Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2),
                Foo(index_field_1=1, index_field_2="baz", non_index_field=7.3),
            ]
        )

        # With pandas >~2.0.3, the following fails with
        # ValueError("Columns with duplicate values are not supported in stack"),
        # instead of SchemaError, via pandera function `reshape_failure_cases`.
        # https://github.com/unionai-oss/pandera/issues/1328
        try:
            FooDataframe(
                [
                    Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2),
                    Foo(index_field_1=1, index_field_2="bar", non_index_field=7.3),
                ]
            )
        except Exception as e:
            assert type(e) in [pa.errors.SchemaError, ValueError]

    def test_getting_and_setting(self) -> None:
        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            index_field_1: Annotated[int, DataframeIndex]
            index_field_2: Annotated[str, DataframeIndex]
            non_index_field: float

        FooDataframe = ObjectsBackingDataframe[Foo]  # noqa: N806

        foo_df = FooDataframe(
            pd.DataFrame(
                [Foo(index_field_1=1, index_field_2="bar", non_index_field=4.2)]
            ).set_index(["index_field_1", "index_field_2"])
        )
        (foo_0,) = list(foo_df)

        assert type(foo_0.index_field_1) is int
        assert foo_0.index_field_1 == 1
        with pytest.raises(SettingOnIndexLevelError):
            foo_0.index_field_1 = 2
        assert type(foo_0.index_field_2) is str
        assert foo_0.index_field_2 == "bar"

        assert foo_0.non_index_field == 4.2
        foo_0.non_index_field = 7.3
        assert foo_0.non_index_field == 7.3
