import typing
import polars as pl


class DataFetch(typing.Protocol):
        def __init__(self, 
                     source: typing.Optional[str],
                     *args, **kwargs) -> None:
                self._source: typing.Optional[str] = source
        
        def fetch_data(self, *args, **kwargs) -> None:
                ...
                
class DataTransform(typing.Protocol):
        def __init__(self,
                     raw_data: typing.Optional[pl.DataFrame],
                     *args, **kwargs) -> None:
                self._raw_data: typing.Optional[pl.DataFrame] = raw_data
                self._processed_data: typing.Optional[pl.DataFrame] = None
                
        def transform_data(self, *args, **kwargs) -> None:
                ...

class DataExport(typing.Protocol):
        def __init__(self,
                     processed_data: typing.Optional[pl.DataFrame],
                     *args, **kwargs) -> None:
                self._processed_data: typing.Optional[pl.DataFrame] = processed_data
                self._exported_data: typing.Optional[pl.DataFrame] = None
        
        def export_data(self, *args, **kwargs) -> None:
                ...


class DataPipeline(typing.Protocol):
        def __init__(self, 
                     fetcher: type[DataFetch],
                     transformer: type[DataTransform],
                     exporter: type[DataExport],
                     *args, **kwargs) -> None:
                self._fetcher: type[DataFetch] = fetcher
                self._transformer: type[DataTransform] = transformer
                self._exporter: type[DataExport] = exporter
        
        def fetch(self, *args, **kwargs) -> 'DataPipeline':
                ...
                
        def transform(self, *args, **kwargs) -> 'DataPipeline':
                ...
                
        def export(self, *args, **kwargs) -> typing.Optional[pl.DataFrame]:
                ...



