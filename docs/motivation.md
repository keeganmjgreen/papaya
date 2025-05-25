# Motivation: The dataframes-versus-dataclasses problem

!!! note

    Skip to the Usage page if more interested in solutions than problems.

Dataframes and dataclasses have their respective advantages and disadvantages. One data structure's strength is another's weakness.

**Dataframes** use dedicated data types (e.g., NumPy arrays) to store, filter, and manipulate data efficiently in terms of both CPU and memory footprint. Pandas' API also enables efficient data analysis. However, dataframes have a number of problems, starting with their data types often being limited, difficult to control, and not seamlessly compatible with Python types:

- Pandas dynamically uses the most efficient data type based on the data a column happens to contain. While this sounds good and lowers the difficulty barrier for new users, it makes data validation difficult. For example, integers are stored in an integer-type column until a `None` value is inserted, at which point the entire column becomes float-typed with the `None` value converted to `NaN`.[^1] These may be the most efficient data types for the column's data, but makes it difficult to get data in the type you expect. Data models should not depend on what kind of data is present.

- Pandas may treat or convert a column as object-typed, in which the column actually stores pointers to the objects' locations in memory. Compared to storing values in a continuous array in memory, this is worse in terms of CPU and memory usage. This is acceptable in Python as it is how Python normally operates, but defeats some purpose of using Pandas. The problem is that this often happens without the user knowing. For example, when a `None` value is added to a boolean-type column, Pandas converts it to object-typed.[^1]

- If a dataframe is empty or columns contain only null values, Pandas does not know what data types they are meant to be, thus falling back to object-typed columns. This 'edge case' is surprisingly common, causes unexpected problems, and often creates overhead for Pandas users for their code to check for zero-length columns or cast their data type.

- Dataframe columns must be referenced by name (string type), which often lacks IDE support for autocomplete and auto-refactoring, making them somewhat error-prone.

- Typical ways of converting between dataframes and dataclasses do not convert between Pandas data types and Python types, yet copy all underlying data at the cost of memory usage. Users should not be getting a `numpy.True_` value when they are expecting `True`.

[^1]: These examples of Pandas' dynamic data type problems can be solved using Pandas' experimental `Int64Dtype` and `BooleanDtype`, but these data types must be used explicitly and manually, adding overhead for Pandas users.

**Dataclasses** are easy-to-define classes for representing entities or data records, with type-annotated data fields that are analogous to a dataframe's columns. A collection (e.g., list) of dataclass instances is analogous to a dataframe. Unlike dataframes, dataclasses cleanly support inheritance, docstrings, properties, and methods to help model the data entity they are meant to represent. However, a collection of dataclass instances often occupies far more memory and is slower to iterate over compared to using and indexing a dataframe.

Papaya allows users to reap the benefits of both dataframes and dataclasses, attempts to minimize their respective weak points, and solves their interoperability problems.
