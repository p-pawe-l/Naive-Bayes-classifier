from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Pipe.kagglehub_pipeline as kh_pp
import polars as pl 
import classifier as cl
import Token.tokenization as tok

data_pipeline: kh_pp.KaggleHubDataPipeline = kh_pp.KaggleHubDataPipeline(
        kh_pp.KaggleHubDataFetch,
        kh_pp.SMS_SPAM_MessageDataTransformer,
        kh_pp.LinearExporter
)

data: pl.DataFrame = data_pipeline.fetch(
        source='uciml/sms-spam-collection-dataset',
        file='spam.csv',
        encoding='latin1'
).transform(
       regex_pattern=r'[?!.,*_&^#$@]',
       class_key='v1',
       text_key='v2',
       drop_columns=['_duplicated_0', '_duplicated_1', ''],
       splitting_character=" ",
       tokenizer=tok.UnigramTokenization, 
).export()



X_normal_split = data['v2_tokenized'].to_list()
X_clean_split = data['v2_cleaned_tokenized'].to_list()
y = data['v1'].to_list()

X_train, X_test, y_train, y_test = train_test_split(X_clean_split, y, test_size=0.08)

classifier: cl.ClassifierNB = cl.ClassifierNB(tokenizer=tok.UnigramTokenization)
classifier.fit(X_train, y_train)

print(classifier.predict(X_test[0]))



