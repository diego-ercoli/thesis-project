from abc import abstractmethod
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_smoothing_spline
import statsmodels.formula.api as smf
from logic.AbstractTimeSeriesAnalyzer import Plot_Type
from logic.TimeSeriesAnalyzer import TimeSeriesAnalyzer


class SplineTimeSeriesAnalyzer(TimeSeriesAnalyzer):
    @abstractmethod
    def forecasting_method(self, x,y,total_range):
        """
        :param x: training examples values
        :param y: target values
        :param total_range: comprising both training and test set x values
        :return: fitting of total range
        """
        pass

    def forecast_trend(self, trend, training_set, test_set):
        """
        :param trend:
        :param training_set:
        :param test_set:
        :return:
        """
        index_train_set = training_set.index
        index_test_set = test_set.index
        index_total = index_train_set.union(index_test_set)
        total_range = np.arange(len(index_total))
        x = np.arange(len(index_train_set))
        y = trend.values
        y_pred = self.forecasting_method(x,y,total_range)
        y_pred = pd.Series(y_pred, index=index_total)
        trend_fc = y_pred.loc[test_set.index]
        if self.plot_type == Plot_Type.FULL:
            ax_trend = trend.plot(figsize=(15,10),color="yellow")
            y_pred.plot(ax=ax_trend, linestyle='--',color="red")
            ax_trend.set_title('Time Plot of Trend forecast using Smoothing spline');
            plt.legend(['trend', 'smoothing_spline']);
            plt.show()
        return trend_fc