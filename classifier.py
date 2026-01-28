import typing
import math
import collections
import Token.tokenization as tokenization
import functools


class ClassifierNB:
        def __init__(self, tokenizer: type[tokenization.TokenizationFunction] = tokenization.UnigramTokenization,
                     smoothing_paramter: float = 1) -> None:
                self._histogram: typing.Dict[typing.Optional[str], typing.Dict] = collections.defaultdict(lambda: collections.defaultdict(int))
                self._classes_probabilities: typing.Dict[typing.Optional[str], float] = None
                self._classes_counter: typing.Dict[typing.Optional[str], int] = None
                self._vocabluary: typing.Optional[typing.Set[str]] = set()
                self._tokenizer: tokenization.TokenizationFunction = tokenizer()
                self._smoothing_parameter: float = smoothing_paramter

        def histogram(self) -> typing.Dict[typing.Optional[str], typing.Dict]: return self._histogram
        def classes_probabilities(self) -> typing.Dict[typing.Optional[str], float]: return self._classes_probabilities
        def classes_counter(self) -> typing.Dict[typing.Optional[str], int]: return self._classes_counter
        
        def _token(func):
                @functools.wraps(func)
                def wrapper(self, text_or_tokens: typing.Union[str, typing.List[str]], *args, **kwargs):
                        if isinstance(text_or_tokens, list):
                                return func(self, text_or_tokens, *args, **kwargs)
                        
                        config = getattr(self, '_kwargs', {})
                        tokens = self._tokenizer.tokenize(text_or_tokens, **config)
                        return func(self, tokens, *args, **kwargs)
                return wrapper
        
        def __calculate_conditional_probability(self, __x, __class) -> float:
                if __class not in self._histogram.keys(): raise KeyError('Cannot fetch data from incorrect class')
                total_in_class: int = sum(self._histogram[__class].values())
                
                if __x not in self._histogram[__class].keys(): return self._smoothing_parameter / (
                        total_in_class + self._smoothing_parameter * len(self._vocabluary))
                else: return (int(self._histogram[__class][__x]) + self._smoothing_parameter) / (
                        total_in_class + self._smoothing_parameter * len(self._vocabluary)
                ) 
                
        def __update_classes_probabilities(self) -> None:
                if self._classes_counter is None: raise ValueError("Cannot update classes probablities without counters")
                total_occurrences: int = sum(self._classes_counter.values())
                self._classes_probabilities = {
                        class_key: (class_occurrence / total_occurrences) for class_key, class_occurrence in self._classes_counter.items()
                }
                
        def __update_vocabluary(self) -> None:
                self._vocabluary = set(list(self._classes_counter.keys()))
        
        def fit(self, X: typing.List, y: typing.List) -> None:
                if len(X) != len(y): raise ValueError("X and y must be equal in size")
                classes_in_data: set = set(y)
                
                if self._classes_counter is None:
                        self._classes_counter = dict.fromkeys(classes_in_data, 0)
                else:
                        for class_label in classes_in_data:
                                if class_label not in list(self._classes_counter.keys()):
                                        self._classes_counter[class_label] = 0

                for probe_index, probe_data in enumerate(X):
                        probe_class: typing.Optional[str] = y[probe_index]
                        self._classes_counter[probe_class] += 1
                        for probe_data_part in probe_data:
                                if probe_class not in list(self._histogram.keys()): 
                                        self._histogram[probe_class][probe_data_part] = 2
                                else:
                                        if probe_data_part in list(self._histogram[probe_class].keys()):
                                                self._histogram[probe_class][probe_data_part] += 1
                                        else:
                                                self._histogram[probe_class][probe_data_part] = 2
                                        
                self.__update_classes_probabilities()
                self.__update_vocabluary()
        
        @_token  
        def predict(self, X: typing.List[typing.Optional[str]]) -> typing.Dict[typing.Optional[str], float]:
                predictions: typing.Dict[typing.Optional[str], float] = {}
                
                for __class, __class_probability in self._classes_probabilities.items(): 
                        log_probablity: float = math.log(__class_probability)
                        for x_i in X:
                                conditional_probability = self.__calculate_conditional_probability(
                                        x_i, __class
                                )
                                log_probablity += math.log(conditional_probability)
                        predictions[__class] = log_probablity
                
                return predictions
                                
                        
                
                
                