import dataclasses

from objects_backing_dataframe import ObjectsBackingDataframe, dataframe_backed_object


def test_readme_example() -> None:
    @dataframe_backed_object
    @dataclasses.dataclass
    class User:
        user_id: int
        name: str
        account_balance: float
        points_balance: int | None

    user_0 = User(user_id=0, name="Wall-E", account_balance=42.0, points_balance=None)
    assert user_0.user_id == 0

    UserDataframe = ObjectsBackingDataframe[User]  # noqa: N806

    user_df = UserDataframe(
        [(1, "Wall-E", 42.0, None)],
        columns=["user_id", "name", "account_balance", "points_balance"],
    )

    user_df.dtypes  # noqa: B018

    users = list(user_df)

    for user in user_df:
        print(user)
        # `User(user_id=0, name="Wall-E", account_balance=42.0, points_balance=None)`
        print(user.name)
        # `'Wall-E'`

    users[0].name = "Burn-E"
    assert user_df.loc[0, "name"] == "Burn-E"
