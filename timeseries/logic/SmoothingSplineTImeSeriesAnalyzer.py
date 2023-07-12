from scipy.interpolate import make_smoothing_spline
from logic.SplineTimeSeriesAnalyzer import SplineTimeSeriesAnalyzer


class SmoothingSplineTimeSeriesAnalyzer(SplineTimeSeriesAnalyzer):
    """
    Compute the (coefficients of) smoothing cubic spline function using lambda to control the tradeoff
    between the amount of smoothness of the curve and its proximity to the data.
    In case lam is None, using the GCV criteria [1] to find it.
    """
    def forecasting_method(self, x,y,total_range):
        #spl is a Bspline object: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html
        spl = make_smoothing_spline(x, y)
        return spl(total_range)

    def get_trend_method(self):
        return "Smoothing spline"