import dataclasses

from objects_convertible_dataframe import DataFrameConvertible


def test_dataframe_convertible() -> None:
    @dataclasses.dataclass
    class User:
        id: int
        name: str

    UserDataFrame = DataFrameConvertible[User]  # noqa: N806

    user_df = UserDataFrame(
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
