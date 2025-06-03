# Usage

Papaya provides an `DataclassesBackingDataframe` class, which acts as a backend for a collection of dataclass instances, and an `DataclassesConvertibleDataframe` class, of which instances can be converted to a collection of dataclass instances. The first of these two is the most useful and important, and so will be featured in most usage examples:

## Objects-Backing Dataframe

Papaya allows you to reuse an existing dataclass definition as the schema for a dataframe, allowing for precise typing and data validation. Start by decorating a dataclass with `papaya.dataframe_backed_object`:

```python
import dataclasses

import papaya as pya

@pya.dataframe_backed_object
@dataclasses.dataclass
class User:
    user_id: int
    name: str
    account_balance: float
    points_balance: int | None
```

Although `User` is now a "dataframe-backed object", its instances are not automatically backed by a dataframe. Instances work normally:

```python
user_0 = User(user_id=0, name="Wall-E", account_balance=42.0, points_balance=None)
user_0.user_id  # `0`
```

Create a subclass of `pandas.DataFrame` for backing `User` instances:

```python
UserDataframe = pya.ObjectsBackingDataframe[User]
```

Instantiate a `UserDataframe` any way that a `pandas.DataFrame` is normally instantiated:

```python
user_df = UserDataframe(
    data=[(1, "Wall-E", 42.0, None)],
    columns=["user_id", "name", "account_balance", "points_balance"],
)
```

When such an `ObjectsBackingDataframe` is instantiated, it will self-validate using [Pandera](https://pandera.readthedocs.io/en/stable/), ensuring that its pandas data types align with the type annotations of its defining dataclass (in this case, `User`):

```python
user_df.dtypes
# user_id              int64
# name                object
# account_balance    float64
# points_balance       Int64 (1)
```

1.  The `points_balance` column is stored using pandas' `Int64` data type, instead of `int64`, to support null values, as `points_balance` can be `None`.

If the data types are incompatible with the dataclass's type annotations, the validation will fail and raise a Pandera `SchemaError`. An `ObjectsBackingDataframe` can be validated at any time by calling its `validate()` method.

`user_df` can be used like a normal `pandas.DataFrame`, except `list(user_df)` returns a list of `User` instances (rather than returning the dataframe's column names), and iterating over `user_df` iterates over `User` instances (rather than iterating over the dataframe's column names):

```python
users = list(user_df)

for user in user_df:
    print(user)  # `User(user_id=0, name="Wall-E", account_balance=42.0, points_balance=None)`
    print(user.name)  # `"Wall-E"`
```

These `users` are backed by `user_df` (even if `user_df` is deleted or goes out of scope).

Papaya's most important feature is that `user[0].name` is not merely *equal to* user 0's name in `user_df` – it *is* user 0's name in `user_df`. If `user[0].name` is updated, so is user 0's name in the dataframe, without requiring data copying or syncing of any kind:

```python
users[0].name = "Burn-E"
user_df.loc[0, "name"]  # `"Burn-E"`
```

> Note: `pandas.DataFrame`s can be instantiated from dataclass instances, and `ObjectsBackingDataframe`s are no exception, e.g.:
>
> ```python
> user_df = UserDataframe(users := [user_0])
> ```
>
> For a normal `pandas.DataFrame`, the `columns=...` should be specified in case `users` is an empty list and thus pandas cannot infer the column names. However, this does not apply for an `ObjectsBackingDataframe`.
>
> !!! warning
>     Here, `list(user_df)[0]` will be a different `User` instance than the `user_0` used to instantiate `user_df`.

## Objects-Convertible Dataframe

TODO

## Special case: datetimes/timestamps

The data type of a pandas datetime/timestamp column incorporates the time zone (or lack thereof) of the column's data. This essentially requires all datetime/timestamp values therein to have the same time zone. It is impossible to mix time zones in a datetime column.

This is good design both in terms of:

- Memory usage: The time zone info is stored only once rather than for each value.

- Data validation: It is good practice to store all timestamps in UTC and convert to a relevant local time zone only when necessary. If a datetime column stores data for only one locale, which is absolutely possible depending on the application, it is acceptable to use the local time zone.

However, individual `datetime` objects do not support enforcing a specific time zone, or enforcing time zone awareness at all, because a `datetime` type annotation does not include the time zone. This is not just generally unfortunate; it makes it difficult to convert between a DataFrame-backed dataclass and its backing DataFrame when the dataclass has a `datetime`-type field. The corresponding column in the DataFrame can only store one time zone -- which time zone should it store? The time zone cannot be reliably inferred from the individual `datetime` values, as they could have a mix of time zones.

The solution is to annotate `datetime.datetime` (or `pandas.Timestamp`) fields with a pandera `DateTime` object that specifies whether the field's values have a time zone and, if they do, which it is:

```python
TODO
```

Failure to do this will result in the following error:

```
TODO
```
