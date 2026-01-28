import polars
import Pipe.pipeline as pipeline
import kagglehub
import typing
import Token.tokenization as tokenization


class KaggleHubDataFetch(pipeline.DataFetch):
        def __init__(self, 
                     source: typing.Optional[str],
                     file: typing.Optional[str],
                     encoding: typing.Optional[str],
                     *args, **kwargs) -> None:
                super().__init__(source, *args, **kwargs)
                self._file: typing.Optional[str] = file
                self._encoding: typing.Optional[str] = encoding

        def fetch_data(self, *args, **kwargs) -> typing.Optional[polars.DataFrame]:
                full_path = kagglehub.dataset_download(self._source)
                lf = polars.read_csv(f'{full_path}/{self._file}', encoding=self._encoding, *args, **kwargs)
                return lf


class SMS_SPAM_MessageDataTransformer(pipeline.DataTransform):
        def __init__(self,
                     raw_data: typing.Optional[polars.DataFrame],
                     regex_pattern: typing.Optional[str],
                     class_key: typing.Optional[str],
                     text_key: typing.Optional[str],
                     drop_columns: typing.List[typing.Optional[str]] = None,
                     skip_words: typing.List[typing.Optional[str]] = None,
                     splitting_character: typing.Optional[str] = " ",
                     tokenizer: type[tokenization.TokenizationFunction] = tokenization.UnigramTokenization,
                     *args, **kwargs) -> None:
                super().__init__(raw_data, *args, **kwargs)
                self._pattern: typing.Optional[str] = regex_pattern
                self._class_key: typing.Optional[str] = class_key
                self._text_key: typing.Optional[str] = text_key
                self._drop_columns: typing.List[typing.Optional[str]] = drop_columns
                self._skip_words: typing.List[typing.Optional[str]] = skip_words
                self._splitting_character: typing.Optional[str] = splitting_character
                self._tokenizer: type[tokenization.TokenizationFunction] = tokenizer
                
                self._cleaned_allias: str = 'cleaned'
                self._tokenized_allias: str = 'tokenized'
                self._filtered_allias: str = 'filtered'
                
        def transform_data(self, *args, **kwargs) -> typing.Optional[polars.DataFrame]:
                if self._raw_data.is_empty(): raise ValueError("Cannot fetch data for transformation")
                """
                ---------------------------------
                Adding column with clean data
                ---------------------------------
                
                Cleaning data from regex patterns to fetch single words 
                instead of punctuation marks 
                """
                transformed_data = self._raw_data.with_columns([
                        polars.col(self._text_key).str.replace_all(self._pattern, "").alias(
                                f'{self._text_key}_{self._cleaned_allias}'
                        )
                ])
                """
                ---------------------------------
                Tokenization | Splitting text
                ---------------------------------
                
                Creating tokenized columns with raw and cleaned data
                """
                transformed_data = transformed_data.with_columns([
                        # Raw data tokenization
                        polars.col(f'{self._text_key}')
                        .map_elements(lambda text_k: self._tokenizer.tokenize(text_k, sep=self._splitting_character, **kwargs), 
                                      return_dtype=self._tokenizer.get_dtype())
                        .alias(f'{self._text_key}_{self._tokenized_allias}'),
                        
                        # Cleaned data tokenization
                        polars.col(f'{self._text_key}_cleaned')
                        .map_elements(lambda text_k: self._tokenizer.tokenize(text_k, sep=self._splitting_character, **kwargs), 
                                      return_dtype=self._tokenizer.get_dtype())
                        .alias(f'{self._text_key}_{self._cleaned_allias}_{self._tokenized_allias}')
                ])
                  
                """
                ---------------------------------
                Dropping provided columns
                ---------------------------------
                
                Removing columns that user consider as irrelevant
                """      
                if self._drop_columns:
                        final_columns = [
                                col for col in transformed_data.collect_schema().names() if col not in self._drop_columns
                        ]
                        transformed_data = transformed_data.select(final_columns)
                
                """
                ---------------------------------
                Filtering skip words
                ---------------------------------
                
                Removing words that user consider as irrelevant
                """  
                if self._skip_words:
                        skip_set: set = set(self._skip_words)
                        filter_token = lambda tokens: [token for token in tokens if token not in skip_set]
                        transformed_data = transformed_data.with_columns([
                        # Raw data filtering
                        polars.col(f'{self._text_key}')
                        .map_elements(filter_token, return_dtype=polars.List(polars.String))
                        .alias(f'{self._text_key}_{self._tokenized_allias}_{self._cleaned_allias}'),
                        
                        # Cleaned data filtering
                        polars.col(f'{self._text_key}_cleanded')
                        .map_elements(filter_token, return_dtype=polars.List(polars.String))
                        .alias(f'{self._text_key}_{self._cleaned_allias}_{self._tokenized_allias}_{self._filtered_allias}')
                ])
                        
                return transformed_data
                        
                        
class LinearExporter(pipeline.DataExport):
        def __init__(self,
                     processed_data: typing.Optional[polars.DataFrame],
                     *args, **kwargs) -> None:
                super().__init__(processed_data, *args, **kwargs)
        
        def export_data(self, *args, **kwargs) -> typing.Optional[polars.DataFrame]:
                return self._processed_data       
        
                
                
        
class KaggleHubDataPipeline(pipeline.DataPipeline):
        def __init__(self,
                     data_fetcher: type[pipeline.DataFetch],
                     data_transformer: type[pipeline.DataTransform],
                     data_exporter: type[pipeline.DataExport],
                     *args, **kwargs) -> None:
                super().__init__(data_fetcher, data_transformer, data_exporter, *args, **kwargs)
                
                self._raw_data: typing.Optional[polars.DataFrame] = None
                self._transformed_data: typing.Optional[polars.DataFrame] = None
                
                self._fetcher_object: typing.Optional[pipeline.DataFetch] = None
                self._transformer_object: typing.Optional[pipeline.DataTransform] = None 
                self._exporter_object: typing.Optional[pipeline.DataExport] = None
        
        def fetch(self,
                  source: typing.Optional[str], file: typing.Optional[str], encoding: typing.Optional[str],
                  *args, **kwargs) -> 'KaggleHubDataPipeline':
                if self._fetcher_object is None:
                        self._fetcher_object = self._fetcher(source, file, encoding, *args, **kwargs)
                self._raw_data = self._fetcher_object.fetch_data()
                
                return self
        
        def transform(self, 
                      *args, **kwargs) -> 'KaggleHubDataPipeline':
                if self._transformer_object is None:
                        self._transformer_object = self._transformer(self._raw_data, *args, **kwargs)
                self._transformed_data = self._transformer_object.transform_data(*args, **kwargs)
                
                return self
        
        def export(self, *args, **kwargs) -> typing.Optional[polars.DataFrame]:
                if self._exporter_object is None:
                        self._exporter_object = self._exporter(self._transformed_data, *args, **kwargs)
                return self._exporter_object.export_data()
                
                
        