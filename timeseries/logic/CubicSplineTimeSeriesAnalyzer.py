import pandas as pd
from patsy.highlevel import dmatrix
from sklearn.linear_model import LinearRegression
from logic.SplineTimeSeriesAnalyzer import SplineTimeSeriesAnalyzer


class CubicSplineTimeSeriesAnalyzer(SplineTimeSeriesAnalyzer):

    def forecasting_method(self, x, y, total_range):
        data = {'x':x,'y':y}
        df = pd.DataFrame(data)
        #NOTE ABOUT DEGREE OF FREEDOM for cubic spline:
        # df = knots + 4 => knots = df - 4 = 6 - 4 = 2

        input_fit = dmatrix('bs(x, df=6, include_intercept=True)',
                            data=df,
                            return_type='dataframe')

        input_prediction = dmatrix('bs(x, df=6, include_intercept=True)',
                                   {'x': total_range.tolist()},
                                   return_type='dataframe')

        spline_reg2 = LinearRegression(fit_intercept=False)
        spline_reg2.fit(input_fit, df.y)
        spline_pred2 = spline_reg2.predict(input_prediction)
        return spline_pred2

    def get_trend_method(self):
        return "Cubic spline"

#%%
