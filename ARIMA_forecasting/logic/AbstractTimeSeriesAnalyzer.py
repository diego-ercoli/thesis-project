from abc import ABC, abstractmethod
from enum import Enum

from matplotlib import pyplot as plt


# %%
class Plot_Type(Enum):
    NONE = 1
    MINIMAL = 2
    FULL = 3


class AbstractTimeSeriesAnalyzer(ABC):
    """
    def __init__(self, plot_type, loss_fun, avoid_negative_values = True ):
        self.plot_type = plot_type
        self.loss_fun = loss_fun
        self.avoid_negative_values = avoid_negative_values
    """
    @abstractmethod
    def split_data(self, df):
        pass

    @abstractmethod
    def decompose(self, training_set):
        pass

    @abstractmethod
    def forecast_residual(self, resid):
        pass

    @abstractmethod
    def extract_seasonality(self, seasonal, test_set):
        pass

    @abstractmethod
    def forecast_trend(self, trend, training_set, test_set):
        pass

    @abstractmethod
    def get_trend_method(self):
        pass

    def __str__(self):
        post = "using POST-SMOOTHING" if self.post_smooth else ""
        return "TS " + post + " with trend method: " + self.get_trend_method()


    def pipeline(self, df, plot_type):
        self.plot_type = plot_type
        #---PIPELINE_DEFINITION---#
        training_set, test_set = self.split_data(df)
        resid, seasonal, trend = self.decompose(training_set)
        remainder_fc = self.forecast_residual(resid, test_set)
        season_period = self.extract_seasonality(seasonal, test_set)
        trend_fc = self.forecast_trend(trend,  training_set, test_set)
        forecast = remainder_fc + trend_fc + season_period
        #---PLOTTING-----#
        if self.avoid_negative_values:
            forecast[forecast < 0] = 0
        error = round(self.loss_fun(true=test_set['freq'], pred=forecast), 2)
        if self.plot_type == Plot_Type.FULL:
            ax_fc = df.plot(style='-', alpha=0.6, figsize=(15,10))
            forecast.plot(style='-',ax=ax_fc)
            plt.legend(['original_data', 'forecast']);
            plt.show()
        if self.plot_type == Plot_Type.FULL or self.plot_type == Plot_Type.MINIMAL:
            ax_tot = test_set.plot(style='-', alpha=0.6, figsize=(15,10))
            forecast.plot(style='-',ax=ax_tot)
            plt.legend(['test_set', 'forecast']);
            plt.show()
            print(f'{self.loss_fun.__name__} Error: {error}')
        return error

