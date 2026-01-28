import typing
import polars

word = typing.Optional[str]
token = word | typing.List[typing.Optional[word]]

class TokenizationFunction(typing.Protocol):
        @staticmethod
        def tokenize(text: typing.Optional[str], *args, **kwargs) -> typing.List[token]:
                ...
        @staticmethod
        def get_dtype() -> polars.DataType:
                ... 
                
                
class UnigramTokenization(TokenizationFunction):
        @staticmethod
        def tokenize(text: typing.Optional[str], *args, **kwargs) -> typing.List[token]:
                split_char: str = kwargs.get('sep', ' ')
                return text.split(sep=split_char)
        @staticmethod
        def get_dtype() -> polars.DataType: return polars.List(polars.String)
        
class BigramTokenization(TokenizationFunction):
        @staticmethod
        def tokenize(text: typing.Optional[str], *args, **kwargs) -> typing.List[token]:
                split_char: str = kwargs.get('sep', ' ')
                glue_char = kwargs.get('glue_char', ' ')
                splitted_text: typing.List[word] = list(text.split(sep=split_char))
                number_of_words = len(splitted_text)
                tokens_collections: typing.List[token] = []
                
                for word_index in range(number_of_words - 1):
                        bigram_token: token = f'{splitted_text[word_index]}{glue_char}{splitted_text[word_index + 1]}'
                        tokens_collections.append(bigram_token)
                
                return tokens_collections
        @staticmethod
        def get_dtype() -> polars.DataType: return polars.List(polars.String)
                
                
                
        


