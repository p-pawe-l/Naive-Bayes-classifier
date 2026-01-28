import Pipe.kagglehub_pipeline as kh_pp
import polars as pl
import Token.tokenization as tok
import Images.plot_distributions as plt_ditro
import Images.plot_math_statistics as plt_math
import Images.plot_experiments as plt_exp

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

distro_plotter: plt_ditro.ClassDistributionPotter = plt_ditro.ClassDistributionPotter(data['v1'], 'Images/Photos')
math_plotter: plt_math.MathStatisticsPlotter = plt_math.MathStatisticsPlotter(data, 'Images/Photos')
experiments_plotter: plt_exp.ExperimentPlotter = plt_exp.ExperimentPlotter(data, 'Images/Photos')
 

if __name__ == "__main__":
        distro_plotter.draw_count_plot(saving_plot=True)
        distro_plotter.draw_circle_plot(saving_plot=True)
        
        math_plotter.draw_basic_statistics_plot(saving_plot=True)
        math_plotter.draw_variance_std_plot(saving_plot=True)
        
        experiments_plotter.run_clean_vs_normal_comparison(test_size=0.2)
        experiments_plotter.run_tokenizer_comparison(test_size=0.2)
        experiments_plotter.generate_confusion_matrix(test_size=0.2)
        experiments_plotter.run_test_size_comparison()
        
        experiments_plotter.draw_training_time_comparison_plot(saving_plot=True)
        experiments_plotter.draw_tokenizer_comparison_plot(saving_plot=True)
        experiments_plotter.draw_test_size_comparison_plot(saving_plot=True)
        experiments_plotter.draw_clean_vs_normal_plot(saving_plot=True)
        experiments_plotter.draw_confusion_matrix_plot(saving_plot=True)