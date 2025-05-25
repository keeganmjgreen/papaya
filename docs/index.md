<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
</style>

<p align="center">
  <img src="assets/papaya_logo_light.png" alt="PAPAYA" width=50% />
</p>

Papaya is a Python library providing precise and efficient interoperability between pandas dataframes and Python dataclasses, including data type validation.

Papaya enables working with both dataframe and dataclasses representations, achieving the benefits of both **without having to duplicate or copy any data between the two data structures**.

Papaya allows you to use a pandas dataframe in one moment and switch to dataclass instances in another, with the dataclass instances using the same dataframe as a backend for storing and retrieving data when their attributes are accessed or updated.

Thanks to pandas dataframes' extremely efficient use of their own backends (predominantly NumPy), this allows a collection of dataclass instances to have a much smaller memory footprint than they otherwise would.
