# Special case: datetimes/timestamps

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
