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
    dataframe_backed_object,
)


def test_dataframe_backed() -> None:
    @dataframe_backed_object
    @dataclasses.dataclass
    class User:
        id: int
        name: str

    UserDataframe = ObjectsBackingDataframe[User]

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
    # Values stored as, for example, `np.int64` in the backing dataframe are cast to `int` when
    #     accessed as instance attributes:
    assert isinstance(user_1.id, int)

    # Updating an attribute updates the corresponding value in the `UserDataFrame` instance:
    user_1.name = "c"
    assert user_df.loc[0, "name"] == "c"

    # The `User` instances reference a shared `UserDataFrame` instance:
    assert isinstance(user_1.__df, pd.DataFrame)
    assert user_1.__df is user_2.__df
    assert user_1.__df is user_df

    # Even if the variable `user_df` is deleted, the `UserDataFrame` instance is still in memory
    #     and accessible by `User` instances:
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
        pandas_timestamp_w_tzinfo_field: Annotated[
            pd.Timestamp, DateTime(tz=ZoneInfo("America/Toronto"))
        ]
        pandas_timestamp_wo_tzinfo_field: Annotated[pd.Timestamp, DateTime(tz=None)]
        pydatetime_w_tzinfo_field: Annotated[
            dt.datetime, DateTime(tz=ZoneInfo("America/Toronto"))
        ]
        pydatetime_wo_tzinfo_field: Annotated[dt.datetime, DateTime(tz=None)]
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
        pandas_timestamp_w_tzinfo_field=pd.Timestamp(
            "2000-04-02 23:59", tz="America/Toronto"
        ),
        pandas_timestamp_wo_tzinfo_field=pd.Timestamp("2000-04-02 23:59"),
        pydatetime_w_tzinfo_field=dt.datetime(
            2000, 4, 2, 23, 59, tzinfo=ZoneInfo("America/Toronto")
        ),
        pydatetime_wo_tzinfo_field=dt.datetime(2000, 4, 2),
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
        _check_foo_instance(foo_0)

    @staticmethod
    def _check_dtypes(df: ObjectsBackingDataframe) -> None:
        df.validate()
        assert len(df.dtypes) == 16
        assert df.dtypes["bool_field"] == np.dtype("bool")
        assert df.dtypes["date_field"] == np.dtype("O")
        assert df.dtypes["enum_field"] == np.dtype("O")
        assert df.dtypes["float_field"] == np.dtype("float64")
        assert df.dtypes["int_field"] == np.dtype("int64")
        assert df.dtypes["list_field"] == np.dtype("O")
        assert df.dtypes["literal_field"] == np.dtype("int64")
        assert df.dtypes["pandas_timestamp_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pandas_timestamp_wo_tzinfo_field"] == np.dtype("<M8[ns]")
        assert df.dtypes["pydatetime_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pydatetime_wo_tzinfo_field"] == np.dtype("<M8[ns]")
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
        pandas_timestamp_w_tzinfo_field=pd.Timestamp(
            "2000-04-02 23:59", tz="America/Toronto"
        ),
        pandas_timestamp_wo_tzinfo_field=pd.Timestamp("2000-04-02 23:59"),
        pydatetime_w_tzinfo_field=dt.datetime(
            2000, 4, 2, 23, 59, tzinfo=ZoneInfo("America/Toronto")
        ),
        pydatetime_wo_tzinfo_field=dt.datetime(2000, 4, 2),
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
        _check_foo_instance(foo_0)

    def test_with_some_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.foo_instance,
                self.Foo(),  # All `None`s.
            ]
        )
        self._check_dtypes(foo_df)
        foo_0, foo_1 = list(foo_df)
        _check_foo_instance(foo_0)
        for field in dataclasses.fields(self.Foo):
            assert getattr(foo_1, field.name) is None

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

    @staticmethod
    def _check_dtypes(df: ObjectsBackingDataframe) -> None:
        df.validate()
        assert len(df.dtypes) == 16
        assert df.dtypes["bool_field"] == pd.BooleanDtype()
        assert df.dtypes["date_field"] == np.dtype("O")
        assert df.dtypes["float_field"] == np.dtype("float64")
        assert df.dtypes["int_field"] == pd.Int64Dtype()
        assert df.dtypes["list_field"] == np.dtype("O")
        assert df.dtypes["literal_field"] == pd.Int64Dtype()
        assert df.dtypes["pandas_timestamp_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pandas_timestamp_wo_tzinfo_field"] == np.dtype("<M8[ns]")
        assert df.dtypes["pydatetime_w_tzinfo_field"] == pd.DatetimeTZDtype(
            tz=ZoneInfo("America/Toronto")
        )
        assert df.dtypes["pydatetime_wo_tzinfo_field"] == np.dtype("<M8[ns]")
        assert df.dtypes["str_field"] == np.dtype("O")
        assert df.dtypes["time_w_tzinfo_field"] == np.dtype("O")
        assert df.dtypes["time_wo_tzinfo_field"] == np.dtype("O")
        assert df.dtypes["uuid_field"] == np.dtype("O")
        assert df.dtypes["zone_info_field"] == np.dtype("O")


class TestCompatibilityWithEnums:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        enum_field: AnEnum

    foo_instance = Foo(enum_field=AnEnum.A)

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_storing_enum_members_as_members(self) -> None:
        foo_df = self.FooDataframe([self.foo_instance])
        foo_df.store_enum_members_as = "members"
        foo_df.validate()
        assert foo_df.dtypes["enum_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "enum_field"]) is AnEnum
        assert foo_df.loc[0, "enum_field"] is AnEnum.A
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        foo_df = self.FooDataframe([self.Foo(enum_field="<invalid-value>")])
        with pytest.raises(pa.errors.SchemaError):
            foo_df.validate()

    def test_storing_enum_members_as_names(self) -> None:
        foo_df = self.FooDataframe([self.foo_instance])
        foo_df.store_enum_members_as = "names"
        foo_df.validate()
        assert foo_df.dtypes["enum_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "enum_field"]) is str
        assert foo_df.loc[0, "enum_field"] == "A"
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        foo_df = self.FooDataframe([self.Foo(enum_field="<invalid-value>")])
        with pytest.raises(pa.errors.SchemaError):
            foo_df.validate()

    def test_storing_enum_members_as_integer_values(self) -> None:
        foo_df = self.FooDataframe([self.foo_instance])
        foo_df.store_enum_members_as = "values"
        foo_df.validate()
        assert foo_df.dtypes["enum_field"] == np.dtype("int64")
        assert type(foo_df.loc[0, "enum_field"]) is np.int64
        assert foo_df.loc[0, "enum_field"] == 1
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        foo_df = self.FooDataframe([self.Foo(enum_field="<invalid-value>")])
        with pytest.raises(pa.errors.SchemaError):
            foo_df.validate()

    def test_storing_enum_members_as_string_values(self) -> None:
        class AnEnum(Enum):
            A = "a"
            B = "b"
            C = "c"

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            enum_field: AnEnum

        FooDataframe = ObjectsBackingDataframe[Foo]
        foo_df = FooDataframe([Foo(enum_field=AnEnum.A)])
        foo_df.store_enum_members_as = "values"
        foo_df.validate()
        assert foo_df.dtypes["enum_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "enum_field"]) is str
        assert foo_df.loc[0, "enum_field"] == "a"
        (foo_0,) = list(foo_df)
        assert type(foo_0.enum_field) is AnEnum
        assert foo_0.enum_field is AnEnum.A

        foo_df = self.FooDataframe([self.Foo(enum_field="<invalid-value>")])
        with pytest.raises(pa.errors.SchemaError):
            foo_df.validate()

    def test_storing_enum_members_as_object_values(self) -> None:
        class AnEnum(Enum):
            A = ZoneInfo("America/Toronto")
            B = ZoneInfo("Asia/Kolkata")
            C = ZoneInfo("Asia/Tokyo")

        @dataframe_backed_object
        @dataclasses.dataclass
        class Foo:
            enum_field: AnEnum

        FooDataframe = ObjectsBackingDataframe[Foo]
        foo_df = FooDataframe([Foo(enum_field=AnEnum.A)])
        foo_df.store_enum_members_as = "values"
        foo_df.validate()
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
        foo_df.validate()
        assert foo_df.dtypes["literal_field"] == np.dtype("O")
        assert type(foo_df.loc[0, "literal_field"]) is int
        assert foo_df.loc[0, "literal_field"] == 1
        (foo_0,) = list(foo_df)
        assert type(foo_0.literal_field) is int
        assert foo_0.literal_field == 1

    def test_validating_literal_values(self) -> None:
        foo_df = self.FooDataframe([self.Foo(literal_field="<invalid-value>")])
        with pytest.raises(pa.errors.SchemaError):
            foo_df.validate()


def _check_foo_instance(foo_instance: type) -> None:
    assert type(foo_instance.bool_field) is bool
    assert foo_instance.bool_field is True
    assert type(foo_instance.date_field) is dt.date
    assert foo_instance.date_field == dt.date(2000, 4, 2)
    assert type(foo_instance.enum_field) is AnEnum
    assert foo_instance.enum_field is AnEnum.A
    assert type(foo_instance.float_field) is float
    assert foo_instance.float_field == 4.2
    assert type(foo_instance.int_field) is int
    assert foo_instance.int_field == 42
    assert type(foo_instance.list_field) is list
    assert foo_instance.list_field == [4, 2]
    assert type(foo_instance.pandas_timestamp_w_tzinfo_field) is pd.Timestamp
    assert foo_instance.pandas_timestamp_w_tzinfo_field == pd.Timestamp(
        "2000-04-02 23:59", tz="America/Toronto"
    )
    assert type(foo_instance.pandas_timestamp_wo_tzinfo_field) is pd.Timestamp
    assert foo_instance.pandas_timestamp_wo_tzinfo_field == pd.Timestamp(
        "2000-04-02 23:59"
    )
    assert type(foo_instance.pydatetime_w_tzinfo_field) is dt.datetime
    assert foo_instance.pydatetime_w_tzinfo_field == dt.datetime(
        2000, 4, 2, 23, 59, tzinfo=ZoneInfo("America/Toronto")
    )
    assert type(foo_instance.pydatetime_wo_tzinfo_field) is dt.datetime
    assert foo_instance.pydatetime_wo_tzinfo_field == dt.datetime(2000, 4, 2)
    assert type(foo_instance.str_field) is str
    assert foo_instance.str_field == "bar"
    assert type(foo_instance.time_w_tzinfo_field) is dt.time
    assert foo_instance.time_w_tzinfo_field == dt.time(
        23, 59, tzinfo=ZoneInfo("America/Toronto")
    )
    assert type(foo_instance.time_wo_tzinfo_field) is dt.time
    assert foo_instance.time_wo_tzinfo_field == dt.time(23, 59)
    assert type(foo_instance.uuid_field) is UUID
    assert type(foo_instance.zone_info_field) is ZoneInfo
    assert foo_instance.zone_info_field == ZoneInfo("America/Toronto")


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

        FooDataframe = ObjectsBackingDataframe[Foo]
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
            # ^ `Field.type` is actually just `int`, not `Union[int]`, so this is never an issue.

        FooDataframe = ObjectsBackingDataframe[Foo]
        foo_df = FooDataframe([Foo(single_union_field=1)])
        (foo_0,) = list(foo_df)
        assert foo_0.single_union_field == 1


class TestStoringNullableIntsAsFloats:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        int_field: int
        nullable_int_field: int | None

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_with_no_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(int_field=1, nullable_int_field=2),
            ]
        )
        foo_df.store_nullable_ints_as_floats = True
        foo_df.validate()
        assert foo_df.dtypes["int_field"] == np.dtype("int64")
        assert foo_df.dtypes["nullable_int_field"] == np.dtype("float64")

    def test_with_some_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(int_field=1, nullable_int_field=None),
                self.Foo(int_field=1, nullable_int_field=2),
            ]
        )
        foo_df.store_nullable_ints_as_floats = True
        foo_df.validate()
        assert foo_df.dtypes["int_field"] == np.dtype("int64")
        assert foo_df.dtypes["nullable_int_field"] == np.dtype("float64")

    def test_with_all_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(int_field=1, nullable_int_field=None),
            ],
        )
        foo_df.store_nullable_ints_as_floats = True
        foo_df.validate()
        assert foo_df.dtypes["int_field"] == np.dtype("int64")
        assert foo_df.dtypes["nullable_int_field"] == np.dtype("float64")


class TestStoringNullableBoolsAsObjects:
    @dataframe_backed_object
    @dataclasses.dataclass
    class Foo:
        bool_field: bool
        nullable_bool_field: bool | None

    FooDataframe = ObjectsBackingDataframe[Foo]

    def test_with_no_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(bool_field=True, nullable_bool_field=True),
            ],
        )
        foo_df.store_nullable_bools_as_objects = True
        foo_df.validate()
        assert foo_df.dtypes["bool_field"] == np.dtype("bool")
        assert foo_df.dtypes["nullable_bool_field"] == np.dtype("O")

    def test_with_some_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(bool_field=True, nullable_bool_field=None),
                self.Foo(bool_field=True, nullable_bool_field=True),
            ],
        )
        foo_df.store_nullable_bools_as_objects = True
        foo_df.validate()
        assert foo_df.dtypes["bool_field"] == np.dtype("bool")
        assert foo_df.dtypes["nullable_bool_field"] == np.dtype("O")

    def test_with_all_nulls(self) -> None:
        foo_df = self.FooDataframe(
            [
                self.Foo(bool_field=True, nullable_bool_field=None),
            ],
        )
        foo_df.store_nullable_bools_as_objects = True
        foo_df.validate()
        assert foo_df.dtypes["bool_field"] == np.dtype("bool")
        assert foo_df.dtypes["nullable_bool_field"] == np.dtype("O")
