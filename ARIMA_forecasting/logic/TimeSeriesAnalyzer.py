from datetime import timedelta, datetime
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stl._stl import STL
import pandas as pd
from logic.AbstractTimeSeriesAnalyzer import AbstractTimeSeriesAnalyzer, Plot_Type


class TimeSeriesAnalyzer(AbstractTimeSeriesAnalyzer):

    def get_trend_method(self):
        return "Linear Regression "

    def __init__(self, loss_fun, avoid_negative_values=True, post_smooth=False):
        self.loss_fun = loss_fun
        self.avoid_negative_values = avoid_negative_values
        self.post_smooth = post_smooth

    def __fill_gaps(self, df1, granularity="D", default_value=0):
        return df1.resample(granularity).asfreq().fillna(default_value)

    def __post_smoothing(self, timeseries):
        """
        Apply loess smoothing
        :param timeseries:
        :return:
        """
        original_x = timeseries.index
        x = np.arange(len(original_x))
        y = timeseries.values
        lowess = sm.nonparametric.lowess
        # span s, which is the proportion of points used to compute the local regression at x
        # 365 * 0.05 = 18
        span = 0.05
        fitted = lowess(y,
                        x,
                        frac=span,
                        xvals=x)
        return pd.Series(fitted, index=original_x)

    def split_data(self, df, data_split=datetime(2022, 1, 1)):
        one_day = timedelta(days=1)
        train, test = df.loc[: data_split - one_day], df[data_split:]
        return self.__fill_gaps(train), self.__fill_gaps(test)

    def decompose(self, training_set):
        stl = STL(
            training_set['freq'],
            period=365,
            seasonal=5,
            robust=True).fit()
        if self.plot_type == Plot_Type.FULL:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(15, 10))
            ax1.plot(stl.observed)
            ax1.set_ylabel('Observed')
            ax2.plot(stl.trend)
            ax2.set_ylabel('Trend')
            ax3.plot(stl.seasonal)
            ax3.set_ylabel('Seasonal')
            ax4.plot(stl.resid)
            ax4.set_ylabel('Residuals')
            plt.show()
        return stl.resid, stl.seasonal, stl.trend

    def forecast_residual(self, resid, test_set):
        auto_model = auto_arima(resid, stepwise=False, trace=False)
        remainder_fc = pd.Series(auto_model.predict(n_periods=test_set.shape[0]).values,
                                 index=test_set.index)
        if self.plot_type == Plot_Type.FULL:
            ax_remainder = resid.plot(style='-', alpha=0.6, figsize=(15, 10))
            remainder_fc.plot(ax=ax_remainder, style='-', title="Forecasting of residual with ARIMA")
            plt.legend(['residual', 'forecast']);
            plt.show()
        return remainder_fc

    def extract_seasonality(self, seasonal, test_set):
        # extract seasonality of previous year
        season_period = seasonal.loc['2021':][:test_set.shape[0]]
        season_period = season_period.set_axis(test_set.index)
        if self.post_smooth:
            post_season_period = self.__post_smoothing(season_period)
        if self.plot_type == Plot_Type.FULL:
            ax_season = season_period.plot(figsize=(15, 10), title='Seasonaly Component in 2021')
            if self.post_smooth:
                post_season_period.plot(ax = ax_season)
            plt.show()
        return post_season_period if self.post_smooth else season_period

    def forecast_trend(self, trend, training_set, test_set):
        """
        LINEAR REGRESSION METHOD
        :param trend:
        :param training_set:
        :param test_set:
        :return:
        """
        index_train_set = training_set.index
        index_test_set = test_set.index
        index_total = index_train_set.union(index_test_set)
        # forecasting the trend with a linear model
        time = np.arange(len(index_train_set))
        X = time.reshape(-1, 1)
        y = trend.values  # target
        model = LinearRegression()
        model.fit(X, y)
        total_range = np.arange(len(index_total))
        y_pred = pd.Series(model.predict(total_range.reshape(-1, 1)), index=index_total)
        trend_fc = y_pred.loc[test_set.index]
        if self.plot_type == Plot_Type.FULL:
            ax_trend = trend.plot(figsize=(15, 10))
            y_pred.plot(ax=ax_trend, linewidth=3)
            ax_trend.set_title('Time Plot of Trend forecast using LINEAR REGRESSION');
            plt.legend(['trend', 'linear_regression']);
            plt.show()
        return trend_fc
