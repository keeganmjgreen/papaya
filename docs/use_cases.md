# Use Cases

Use Papaya if you want the benefits of both **dataclasses** (typed fields with docstrings, properties and methods, subclasses) and **dataframes** (efficient API for data analysis and manipulation, low CPU and memory usage).

## Use case examples

### Proof-of-concept for a database-backed application

In database-backed applications, the database may store lots of data, but the application itself needs to serve, ingest, or process only a small amount of data at a time. In Papaya, the dataclasses-dataframe relationship is much like the objects-database relationship in an ORM (such as SQLAlchemy). Because of this relationship, Papaya can be used as a stand-in for a database in a proof-of-concept for a database-backed application.

Rather than querying the database for the data to work with, a core dataframe can be filtered down and converted to a relatively small quantity of dataclass instances to process:

```python
TODO
```

And rather than executing a database operation to update a record, a new value can be set on a dataclass instance and automatically "committed" to its backing dataframe:

```python
TODO
```

### Simulations for operations research

In operations research, it is common to want to simulate a prescribed operational scenario over time to see how adjustments or optimizations can improve operational performance. This can be used to help develop a software system for optimizing real-life operations.

For example, simulating electric bus charging operations at a bus depot over time, where the simulation is subject to a dataset constraining when the electric buses need to charge and how much. The dataset may include years of data and consume gigabytes of memory if not stored in a dataframe, but the simulation's intent is not to process or use all of it at once, as simulating over only a few months may turn out to be enough to see improved operational performance.

In such a use case, it is extremely helpful to be able to convert dataset records to dataclass instances when needed, and use them to keep the original dataset up-to-date, all of which Papaya enables.

It is also common to want to deploy such a simulation alongside some software-under-test, all running in real-time, to test that the whole system works as intended and that performance is good.

For example, deploying the aforementioned simulation of electric bus charging alongside some software that optimizes the charging operations. Running the optimization software in a real-time deployment, and having a simulation to provide it with the data it needs to run, can help catch bugs before deploying to production.

## Other use cases

### Reading/writing CSVs

Papaya can be used to load a CSV file into a collection of dataclass instances, rather than a dataframe:

```python
TODO
```

Papaya can also be used to write a collection of dataclass instances to a CSV file:

```python
TODO
```

### Unit tests

Papaya can aid in writing and maintaining unit tests by making the creation of dataframes easier, both as test input and expected output:

```python
TODO
```
