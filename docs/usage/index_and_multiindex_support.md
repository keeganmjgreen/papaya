# Index and MultiIndex Support

By default, the Index of a `ObjectsBackingDataframe` is used to uniquely identify rows and associate them with dataframe-backed objects.

However, it is often more convenient to work with dataframes when they have a meaningful Index or MultiIndex. For example, objects often have ID attributes anyway.

## Example: Using a dataclass field in an Index

```python
import dataclasses

import papaya as pya

@pya.dataframe_backed_object
@dataclasses.dataclass
class User:
    user_id: Annotated[int, DataframeIndex]  # (1)!
    name: str

UserDataframe = ObjectsBackingDataframe[User]

user_df = UserDataframe(
    pd.DataFrame(
        [User(user_id=0, name="Wall-E")]
    ).set_index("user_id")  # (2)!
)
```

1.  This specifies that `User.user_id` should be used as the Index.
2.  By default, the Index must be set before instantiating the `ObjectsBackingDataframe`. This can be overridden by setting `PapayaConfig.set_index` to `True` or `"auto"`.

`user_df` can be used as if it were a normal `pandas.DataFrame` with an Index called `"user_id"`. Interoperability between `user_df` and `User` objects is maintained, except that trying to set the index field will raise an `SettingOnIndexLevelError` (as it would change or break the mapping between `user_df` rows and `User` objects).

```python
(user_0,) = list(user_df)

user_0.user_id  # `1`
user_0.name  # `'Wall-E'`

user_0.user_id = 2  # Trying to set on an index field raises a `SettingOnIndexLevelError`.
```

## Example: Using multiple dataclass fields in a MultiIndex

```python
import dataclasses
from uuid import UUID

import papaya as pya

@pya.dataframe_backed_object
@dataclasses.dataclass
class UserAddress:
    user_id: Annotated[int, DataframeIndex]
    address_id: Annotated[UUID, DataframeIndex]
    name: str

UserAddressDataframe = ObjectsBackingDataframe[UserAddress]

user_address_df = UserAddressDataframe(
    pd.DataFrame(
        [UserAddress(user_id=0, address_id=UUID(int=42, version=4), name="The Axiom")]
    ).set_index(["user_id", "address_id"])
)
```

The functionality is essentially identical for a MultiIndex as it is for an Index:

```python
(address_0,) = list(user_df)

address_0.user_id  # `1`
address_0.address_id  # `UUID('00000000-0000-4000-8000-00000000002a')`
address_0.name  # `'The Axiom'`

address_0.user_id = 2  # Trying to set on an index field raises a `SettingOnIndexLevelError`.
```

## Configuration

The behavior when instantiating an `ObjectsBackingDataframe` can be controlled using `PapayaConfig.set_index`:

- `set_index=False` is the default, used in the above examples.

- `set_index=True` requires that index field(s) are in the columns. They will be moved to the [Multi]Index automatically:

    ```python
    @pya.dataframe_backed_object
    @dataclasses.dataclass
    class User:
        user_id: Annotated[int, DataframeIndex]
        name: str

        papaya_config = pya.PapayaConfig(set_index=True)  # (1)!

    UserDataframe = ObjectsBackingDataframe[User]

    user_df = UserDataframe([User(user_id=0, name="Wall-E")])  # (2)!
    ```

    1.  As a result of `set_index=True`, `user_id` will automatically be set as the index of `user_df`.
    2.  Manually calling `.set_index("user_id")` is not necessary.

- `set_index="auto"` moves index field(s) to the [Multi]Index if they are all in the columns.
