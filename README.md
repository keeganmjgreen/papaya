<p align="center">
  <img src="assets/papaya_logo.svg" alt="PAPAYA" />
</p>

Papaya is a Python library providing precise and efficient interoperability between Pandas dataframes and Python dataclasses, including data type validation. Papaya enables working with both dataframe and dataclasses representations, achieving the benefits of both **without having to duplicate or copy any data between the two data structures**. Papaya allows you to use a Pandas dataframe in one moment and switch to dataclass instances in another, with the dataclass instances using the same dataframe as a backend for storing and retrieving data when their attributes are accessed or updated. Thanks to Pandas dataframes' extremely efficient use of their own backends (predominantly NumPy), this allows a collection of dataclass instances to have a much smaller memory footprint than they otherwise would.

## Motivation: The dataframes-versus-dataclasses problem

> Skip to the Usage section if more interested in solutions than problems.

Dataframes and dataclasses have their respective advantages and disadvantages. One data structure's strength is another's weakness.

**Dataframes** use dedicated data types (e.g., NumPy arrays) to store, filter, and manipulate data efficiently in terms of both CPU and memory footprint. Pandas' API also enables efficient data analysis. However, dataframes have a number of problems, starting with their data types often being limited, difficult to control, and not seamlessly compatible with Python types:

- Pandas dynamically uses the most efficient data type based on the data a column happens to contain. While this sounds good and lowers the difficulty barrier for new users, it makes data validation difficult. For example, integers are stored in an integer-type column until a None value is inserted, at which point the entire column becomes float-typed with the None value converted to NaN.* These may be the most efficient data types for the column's data, but makes it difficult to get data in the type you expect. Data models should not depend on what kind of data is present.

- Pandas may treat or convert a column as object-typed, in which the column actually stores pointers to the objects' locations in memory. Compared to storing values in a continuous array in memory, this is worse in terms of CPU and memory usage. This is acceptable in Python as it is how Python normally operates, but defeats some purpose of using Pandas. The problem is that this often happens without the user knowing. For example, when a None value is added to a boolean-type column, Pandas converts it to object-typed.*

- If a dataframe is empty or columns contain only null values, Pandas does not know what data types they are meant to be, thus falling back to object-typed columns. This 'edge case' is surprisingly common, causes unexpected problems, and often creates overhead for Pandas users for their code to check for zero-length columns or cast their data type.

- Dataframe columns must be referenced by name (string type), which often lacks IDE support for autocomplete and auto-refactoring, making them somewhat error-prone.

- Typical ways of converting between dataframes and dataclasses do not convert between Pandas data types and Python types, yet copy all underlying data at the cost of memory usage. Users should not be getting a `numpy.True_` value when they are expecting `True`.

*Note that these examples of Pandas' dynamic data type problems can be solved using Pandas' experimental `Int64Dtype` and `BooleanDtype`, but these data types must be used explicitly and manually, adding overhead for Pandas users.

**Dataclasses** are easy-to-define classes for representing entities or data records, with type-annotated data fields that are analogous to a dataframe's columns. A collection (e.g., list) of dataclass instances is analogous to a dataframe. Unlike dataframes, dataclasses cleanly support inheritance, docstrings, properties, and methods to help model the data entity they are meant to represent. However, a collection of dataclass instances often occupies far more memory and is slower to iterate over compared to using and indexing a dataframe.

Papaya allows users to reap the benefits of both dataframes and dataclasses, attempts to minimize their respective weak points, and solves their interoperability problems.

## Usage

Papaya provides an `DataclassesBackingDataframe` class, which acts as a backend for a collection of dataclass instances, and an `DataclassesConvertibleDataframe` class, of which instances can be converted to a collection of dataclass instances. The first of these two is the most useful and important, and so will be featured in most usage examples:

### Objects-Backing Dataframe

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
UserDataframe = ObjectsBackingDataframe[User]
```

Instantiate a `UserDataframe` any way that a `pandas.DataFrame` is normally instantiated:

```python
user_df = UserDataframe(
    [(1, "Wall-E", 42.0, None)],
    columns=["user_id", "name", "account_balance", "points_balance"],
)
```

Call `user_df.validate()` to ensure that its Pandas data types align with `User`'s type annotations and validate `user_df`'s existing data against them (uses the [Pandera](https://pandera.readthedocs.io/en/stable/) library):

```python
user_df.validate()
# `points_balance` is converted from `int64` to `Int64` to support null values, as `points_balance` can be `None`:
user_df.dtypes
# user_id              int64
# name                object
# account_balance    float64
# points_balance       Int64
```

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
> ```python
> users = [user_0]
> user_df = UserDataframe(users)
> ```
> For a normal `pandas.DataFrame`, the `columns=...` should be specified in case `users` is an empty list and thus Pandas cannot infer the column names. However, this does not apply for an `ObjectsBackingDataframe`.
> 
> Warning: Here, `list(user_df)[0]` will be a different `User` instance than the `user_0` used to instantiate `user_df`.

### Objects-Convertible Dataframe


## TODOs

- Conversion.
- Indexes.
- Cascading deletes.
- ns/ms
- Categorical
- Filter
- Converting tzs
- Get df / in init
- DF creation on object creation? no
- Property
- fset code?
- fget code in DataclassesConvertibleDataframe
- properties for indexing
- @dataframe_backed_object includes @dataclass?
- check if dataclass in @dataframe_backed_object
- subclass eg timestamp
