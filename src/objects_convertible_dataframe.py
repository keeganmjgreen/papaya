from typing import Iterator

from objects_dataframe_base import ObjectsDataframeBase


class DataFrameConvertible[T](ObjectsDataframeBase):

    def __iter__(self) -> Iterator[T]:
        self.validate()
        for _, row in self.iterrows():
            yield self._dataframe_objects_class(**row.to_dict())
