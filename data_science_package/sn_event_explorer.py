""" Module to calculate anomalies in past or future datapoints with respect to events"""
import json
import math
from urllib.request import urlopen
from typing import List
from pathlib import Path
from datetime import timedelta

import ipywidgets as widgets
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from IPython.display import display, clear_output, HTML
from adtk.detector import ThresholdAD, QuantileAD, InterQuartileRangeAD, PersistAD, LevelShiftAD, \
    VolatilityShiftAD, AutoregressionAD, SeasonalAD, MinClusterDetector, OutlierDetector
from ics import Calendar
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

pio.renderers.default = 'notebook_connected'
init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None  # default='warn'


class Plotter:
    """
    Plotter is used to plot an interactive overview of the calculated data
    """
    def __init__(self):
        selector_buttons = list([
            {"count": 1, "label": "1h", "step": "hour", "stepmode": "backward"},
            {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
            {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
            {"step": 'all'}
        ])
        self.__plot_x_axis = {
            "rangeselector": {"buttons": selector_buttons},
            "rangeslider": {"visible": True},
            "type": 'date'
        }

    def get_x_axis_plot(self):
        """
        required to plot interactive overview from another class
        """
        return self.__plot_x_axis

    def plot_column(self, dataframe, column_to_plot: str):
        """
        Function to plot the selected column

        :param dataframe: dataset in Pandas dataframe format.
                        Requires at least two columns: Time and target_column
        :param column_to_plot: name of the column to be plotted
        :return:
        """
        fig = go.Figure()
        if dataframe.index.name == 'Time':
            dataframe = dataframe.reset_index()
        fig.add_trace(go.Scatter(x=list(dataframe['Time']), y=list(dataframe[column_to_plot])))
        title = 'Time series inspection: ' + column_to_plot
        fig.update_layout(title_text=title, xaxis=self.__plot_x_axis)
        fig.show()


    def plot_anomaly_column(self, data, anomalies, column_to_plot: str, event_table: None):
        """
        :param data: Pandas Dataframe of the dataset
        :param anomalies: Pandas Dataframe of anomalies
        :param column_to_plot: desired column within the dataset
        :param event_table: Pandas Dataframe of the events
        :return: a plot of the detected anomalies
        """
        fig = go.Figure()
        data['is_anomaly'] = anomalies

        fig.add_trace(go.Scatter(name='Measurement', x=list(data.index),
                                 y=list(data[column_to_plot])))
        fig.add_trace(go.Scatter(
            name="Anomaly",
            x=data[data['is_anomaly']].index,
            y=data[data['is_anomaly']][column_to_plot],
            mode='markers',
            marker={"color": 'red', "size": 2, "line": {"color": "red", "width": 2}}
        ))

        if event_table is not None:
            event_table['begin'] = event_table['begin'].dt.tz_localize(None)
            event_table['end'] = event_table['end'].dt.tz_localize(None)
            start = np.datetime64(data.index[0])
            end = np.datetime64(data.index[-1])
            mask = (event_table['begin'] > start) & (event_table['end'] <= end)
            filtered_event_table = event_table.loc[mask]
            color_dict = {}
            v_rect_colors = all_colors_except_red()
            for filtered_event in filtered_event_table.reset_index().values:
                if filtered_event[4] == "Holidays":
                    color_dict[filtered_event[4]] = 'red'
                else:
                    color_dict[filtered_event[4]] = v_rect_colors.pop(0)
            for filtered_event in filtered_event_table.reset_index().values:
                fig.add_vrect(x0=filtered_event[2], x1=filtered_event[3],
                              annotation={"text": filtered_event[1], "textangle": -90},
                              fillcolor=color_dict[filtered_event[4]], opacity=0.3,
                              layer="below", line_width=0)

        fig.update_layout(title_text='Detected Anomalies', xaxis=self.__plot_x_axis)
        fig.show()


class AnomalyDetector:
    """
    selection of anomaly detection algorithms
    """

    def __init__(self):
        self.__target_column = None
        self.__series = None
        self.__anomalies = None

    def get_anomalies(self):
        return self.__anomalies

    def with_data(self, dataframe, target_column: str):
        if dataframe.index.name != 'Time':
            dataframe = dataframe.set_index('Time')
        series = dataframe[[target_column]]
        series.index = pd.to_datetime(series.index, utc=True, unit='ms')
        self.__series = series
        self.__target_column = target_column
        return self

    def apply_threshold(self, lower_threshold=0, upper_threshold=4):
        threshold_ad = ThresholdAD(high=upper_threshold, low=lower_threshold)
        self.__anomalies = threshold_ad.detect(self.__series.copy())
        return self

    def apply_quantile(self, lower_quantile=0.05, upper_quantile=0.95):
        quantile_ad = QuantileAD(high=upper_quantile, low=lower_quantile)
        self.__anomalies = quantile_ad.fit_detect(self.__series.copy())
        return self

    def apply_iqr(self, iq_range=10.0):
        self.__anomalies = InterQuartileRangeAD(c=iq_range).fit_detect(self.__series.copy())
        return self

    def apply_persistAD(self, threshold=20, side='both', window_size=20):
        persist_ad = PersistAD(c=threshold, side=side)
        persist_ad.window = window_size
        self.__anomalies = persist_ad.fit_detect(self.__series.copy())
        return self

    def apply_levelshift(self, threshold=5, side='both', window_size=25):
        level_shift_ad = LevelShiftAD(c=threshold, side=side, window=window_size)
        self.__anomalies = level_shift_ad.fit_detect(self.__series.copy())
        return self

    def apply_volatilityshift(self, threshold=5, side='both', window_size=25):
        volatility_shift_ad = VolatilityShiftAD(c=threshold, side=side, window=window_size)
        self.__anomalies = volatility_shift_ad.fit_detect(self.__series.copy())
        return self

    def apply_localoutlier(self, contamination_value=None):
        if contamination_value:
            detector = LocalOutlierFactor(contamination=contamination_value)
        else:
            detector = LocalOutlierFactor()
        outlier_detector = OutlierDetector(detector)
        self.__anomalies = outlier_detector.fit_detect(self.__series.copy())
        return self

    def apply_seasonality(self, threshold=5):
        seasonal_ad = SeasonalAD(c=threshold, side="both")
        try:
            self.__anomalies = seasonal_ad.fit_detect(self.__series.copy())
        except Exception:
            print('Error: Could not find significant seasonality.')
            return None
        return self

    def apply_autoregression(self, threshold=3.0, n_steps=7 * 2, step_size=10):
        autoregression_ad = AutoregressionAD(n_steps=n_steps, step_size=step_size, c=threshold)
        self.__anomalies = autoregression_ad.fit_detect(self.__series.copy())
        return self

    def apply_kMeans(self, n_clusters=2):
        min_cluster_detector = MinClusterDetector(KMeans(n_clusters=n_clusters))
        self.__anomalies = min_cluster_detector.fit_detect(self.__series.copy())
        return self

    def apply_prophet(self):
        self.__anomalies = calculate_anomalies_with_prophet(self.__series.copy(),
                                                            self.__target_column)
        return self

    def and_plot(self, eventtable=None):
        Plotter().plot_anomaly_column(self.__series.copy(), self.__anomalies,
                                      self.__target_column, eventtable)


def is_during_event(timestamp, start, end):
    """
    return 1 if timestamp is within timeframe between start and end, else return 0
    :return: 1 or 0
    """
    if pd.Timestamp(start) < timestamp < pd.Timestamp(end):
        return 1
    return 0


def is_event(data, event_table):
    """
    :param data:
    :param event_table:
    :return: 1 or 0
    """
    result = [is_during_event(data, row[0], row[1])
              for row in event_table[['begin', 'end']].to_numpy()]
    if max(result) > 0:
        return 1
    return 0


def create_model_input(dataframe, selected_column, event_table):
    """
    Create Dataframe for prophet algorithms

    :param dataframe: Pandas Dataframe of the dataset
    :param selected_column: relevant column within the dataset
    :param event_table: Pandas Dataframe of events
    :return: Pandas Dataframe in the correct format for FBProphet
    """
    if dataframe.index.name == 'Time':
        dataframe = dataframe.reset_index()
    if 'Time' not in dataframe:
        dataframe['Time'] = dataframe.index
    dataframe['Time'] = pd.to_datetime(dataframe['Time'], unit='ms')
    dataframe = dataframe.set_index("Time", drop=False)

    df_model_input = dataframe[['Time', selected_column]] \
        .rename(columns={'Time': 'ds', selected_column: 'y'})
    df_model_input['ds'] = df_model_input['ds'].dt.tz_localize(None)

    if event_table is not None and not event_table.empty:
        for entry in event_table['event'].unique().tolist():
            column_name = 'is_' + entry
            filtered_event_table = event_table.loc[event_table['event'] == entry]
            df_model_input[column_name] = df_model_input['ds'] \
                .apply(is_event, args=(filtered_event_table,))

    return df_model_input


def calculate_anomalies_with_prophet(dataframe, selected_column):
    """
    Use FBProphet algorithm to detect anomalies.
    Dataset entries where the actual value is not within the expected confidence interval
    are considered an anomaly.
    :param dataframe: Pandas Dataframe of the dataset
    :param selected_column: relevant column within the dataset
    :return: detected anomalies
    """
    prophet_series = create_model_input(dataframe, selected_column, event_table=None)
    model = Prophet(interval_width=0.99, yearly_seasonality=False, weekly_seasonality=True)
    model.fit(prophet_series)
    forecast = model.predict(prophet_series)
    performance = pd.merge(
        prophet_series,
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
    )
    performance["anomaly"] = performance.apply(
        lambda rows: 1 if ((rows.y < rows.yhat_lower) | (rows.y > rows.yhat_upper))
        else 0,
        axis=1,
    )
    detected_anomalies = performance[performance["anomaly"] == 1].sort_values(
        by="ds"
    )
    anomalies = (
        prophet_series.set_index("ds")
            .combine_first(detected_anomalies.set_index("ds"))
            .reset_index()
    )
    anomalies[selected_column] = anomalies["anomaly"]
    anomalies["ds"] = anomalies["ds"].dt.tz_localize("UTC")
    anomalies = anomalies.set_index("ds")
    anomalies = anomalies.filter([selected_column])
    anomalies[selected_column] = anomalies[selected_column].apply(
        lambda anom: anom == 1
    )
    anomalies[selected_column].fillna(False, inplace=True)
    anomalies.index.names = ['Time']
    return anomalies


class EventExplorer:
    """
    EventExplorer takes two arguments:
    - Dataframe with interface data
    - Path to directory with event csv-files which need to consist of three columns (event:str,
      begin:datetime, end:datetime). Column names are irrelevant
    """

    def __init__(self, data: pd.DataFrame, target_column: str, path_to_event_files: str,
                 link_collection: dict = None):

        self.plotter = Plotter()

        # Initialize Data
        if data.index.name != 'Time':
            data = data.set_index('Time')
        self.data = data
        self.target_column = target_column

        # Detect and save past anomalies
        self.anomaly_data = pd.DataFrame()
        self.detect_anomalies()

        # Get all event csv-files and concat them to one dataframe
        self.all_event_data = collect_events_from_path(path_to_event_files, link_collection)

        # Crop Events to past data for inspection
        self.past_event_data = crop_data_to_timespan(self.all_event_data, self.data.index[0],
                                                     self.data.index[-1])

        # Define plot height in dependence of longest event name to make sure vrects fit in plot
        self.plot_height = len(self.all_event_data['event'].apply(
            lambda x: (len(x), x)).max()[1]) * 20 + 200

        # Get a color for every event type
        self.event_color_dict = get_event_color_dict(self.all_event_data)

        # Map Events to event type e.g. "Holidays: [Karfreitag, 1.Mai,...]"
        self.event_category_map = self.all_event_data.groupby('eventtype')['event'] \
            .apply(set).to_dict()
        self.all_event_types = self.all_event_data.eventtype.unique().tolist()
        self.checkbox_names = self.past_event_data.eventtype.unique().tolist()

        # Manages Output of inspection
        self.inspect_output = widgets.Output()

        self.anomaly_bar_output = widgets.Output()

        # toggles event names depending on checkbox selection
        self.active_event_names_inspection = []

        # Displays "Processing..." while updating inspect fig
        self.status_label = widgets.Label(value="")

        # Initialize Inspect Figure
        self.inspect_fig = go.Figure()
        self.inspect_fig.add_trace(
            go.Scatter(name=self.target_column, x=list(self.data.index),
                       y=list(self.data[self.target_column])))

        # Add Anomaly dots
        self.inspect_fig.add_trace(go.Scatter(
            name="Unexplained Anomalies",
            x=self.data[self.anomaly_data['unexplained']].index,
            y=self.data[self.anomaly_data['unexplained']][self.target_column],
            mode='markers',
            marker={"color": 'red', "size": 2, "line": {"color": "red", "width": 2}}
        ))

        title = 'Time series inspection: ' + self.target_column
        self.inspect_fig.update_layout(title_text=title, xaxis=self.plotter.get_x_axis_plot(),
                                       yaxis_visible=False, yaxis_showticklabels=False,
                                       height=self.plot_height)

        # initialize checkboxes according to checkbox_names
        self.inspect_checkboxes = []
        for name in self.checkbox_names:
            checkbox = widgets.Checkbox(
                value=False,
                description=name,
                disabled=False,
                indent=False
            )
            self.inspect_checkboxes.append(checkbox)
            checkbox.observe(self.on_inspect_checkbox_change)

        # display checkboxes in three columns
        self.inspect_grid = make_checkbox_grid(self.inspect_checkboxes)

        self.prophet = None

        # Initialize for prediction
        self.predict_output = widgets.Output()
        self.model = None
        self.forecast = None
        self.predict_fig = go.Figure()

        self.forecasting_input = None
        self.converted_holidays = None
        self.filtered_events = None

    def list_all_event_types(self):
        """
        Returns a list of all types of events.
        Used as a feedback to make sure all desired data was loaded.
        """
        return_message = "Loaded the following event types: "
        return_message += ", ".join(self.all_event_types)
        return return_message

    def detect_anomalies(self):
        """
        Apply FBProphet algorithm to detect anomalies in past data
        """
        # get custom defined anomalies
        anomaly_detector = AnomalyDetector().with_data(self.data, self.target_column)
        anomaly_detector.apply_prophet()
        anomaly_data = anomaly_detector.get_anomalies()
        anomaly_data.index = anomaly_data.index.tz_localize(None)
        # add detected anomalies to df
        self.anomaly_data['anomalies'] = anomaly_data
        # everything is unexplained before event inspection
        self.anomaly_data['unexplained'] = anomaly_data

    def inspect_anomalies_with_events(self):
        """
        Main function that is called in the Jupyter notebook.
        Returns a plot of the data with checkboxes for each event type.
        """
        # display column grid with checkboxes
        display(self.inspect_grid)
        # show status "Processing..." when plot is updated
        display(self.status_label)
        # show initial plot
        self.display_inspect_fig()
        # show everything added to self.inspect_output
        display(self.inspect_output)

    def predict_with_selection(self, future_days=7, interval=1,
                               daily=True, weekly=True, yearly=False, seasonality="additive"):
        """
        Prediction of future data points and anomalies according to the selected events.

        :param future_days: number of future days
        :param interval: interval of the prediction frequency (in hours)
        :param daily: calculate forecast with daily seasonality
        :param weekly: calculate forecast with weekly seasonality
        :param yearly: calculate forecast with yearly seasonality
        :param seasonality: calculate forecast with additive or multiplicative seasonality
        """
        self.forecast = None
        future_periods = math.floor(future_days * (24 / interval))
        self.make_forecast(future_periods=future_periods, interval=interval,
                            daily=daily, weekly=weekly, yearly=yearly,
                            seasonality=seasonality)
        self.plot_prediction_with_anomalies_and_events()

        clear_output(wait=True)
        display(self.predict_output)

    def check_fit(self):
        """
        calculate and plot the forecasted and actual values and the residuals between these values
        """
        self.forecast = None
        self.make_forecast()

        past_forecast = self.forecast.loc[self.forecast['ds'] <= self.data.index[-1]]
        past_forecast.set_index('ds', inplace=True)

        past_forecast[self.target_column] = self.data[self.target_column]

        past_forecast['residuals'] = past_forecast[self.target_column] - past_forecast['yhat']

        past_forecast_filtered = past_forecast[['residuals', 'yhat', self.target_column]]

        past_forecast_filtered.plot(subplots=True, figsize=(20, 3), sharey=True)

    def on_inspect_checkbox_change(self, change):
        """
        Handles check / uncheck of checkboxes
        """
        if change['name'] == 'value':
            self.status_label.value = "Processing..."
            self.set_checkboxes_state(self.inspect_checkboxes, True)
            if change['new']:
                self.active_event_names_inspection.append(change['owner'].description)
            else:
                self.active_event_names_inspection.remove(change['owner'].description)
            self.update_inspect_fig()
            self.set_checkboxes_state(self.inspect_checkboxes, False)
            self.status_label.value = ""

    def display_inspect_fig(self):
        """
        display inspect figure
        """
        with self.inspect_output:
            display(self.inspect_fig)

    def display_predict_fig(self):
        """
        display prediction figure
        """
        with self.predict_output:
            display(self.predict_fig)

    def update_inspect_fig(self):
        """
        Update the inspect figure when event checkboxes are changed
        """
        # clear event annotations
        self.clear_vrects(self.inspect_fig)

        # check for active events (checked checkboxes)
        if self.active_event_names_inspection:
            active_events = self.past_event_data[
                self.past_event_data['eventtype'].isin(self.active_event_names_inspection)]

            for event in active_events.reset_index().values:
                self.inspect_fig.add_vrect(x0=event[2], x1=event[3],
                                           annotation={"text": event[1], "textangle": -90},
                                           fillcolor=self.event_color_dict[event[4]],
                                           opacity=0.3, layer="below", line_width=0)

        self.update_anomalies()
        title = 'Time series inspection: ' + self.target_column
        self.inspect_fig.update_layout(title_text=title, xaxis=self.plotter.get_x_axis_plot(),
                                       yaxis_visible=False, yaxis_showticklabels=False,
                                       height=self.plot_height)

        with self.inspect_output:
            # Reset output and show updated plot
            clear_output(wait=True)
            self.display_inspect_fig()

    def update_anomalies(self):
        """
        Update the plot to distinguish between explained and unexplained anomalies
        """
        # check which events are selected
        active_event_data = self.past_event_data[
            self.past_event_data['eventtype'].isin(self.active_event_names_inspection)]

        # get begin-end intervals of events
        event_intervals = pd.IntervalIndex.from_arrays(active_event_data['begin'],
                                                       active_event_data['end'], closed='both')

        index_in_event_interval = self.anomaly_data.index.map(lambda x:
                                                              any(event_intervals.contains(x)))

        # set explained true if in interval
        self.anomaly_data['explained'] = self.anomaly_data['anomalies'] & index_in_event_interval

        # set unexplained false if not in interval
        self.anomaly_data['unexplained'] = self.anomaly_data['anomalies'] \
                                           & ~self.anomaly_data['explained']

        # just update when already created
        if len(self.inspect_fig.data) == 3:
            # update traces
            self.inspect_fig.data[1].x = self.data[self.anomaly_data['unexplained']].index
            self.inspect_fig.data[1].y = \
                self.data[self.anomaly_data['unexplained']][self.target_column]

            self.inspect_fig.data[2].x = self.data[self.anomaly_data['explained']].index
            self.inspect_fig.data[2].y = \
                self.data[self.anomaly_data['explained']][self.target_column]

        # create trace of explained anomalies
        elif len(self.inspect_fig.data) == 2:
            # initialise explanation trace
            self.inspect_fig.add_trace(go.Scatter(
                name="Explained Anomalies",
                x=self.data[self.anomaly_data['explained']].index,
                y=self.data[self.anomaly_data['explained']][self.target_column],
                mode='markers',
                marker={"color": 'green', "size": 2, "line": {"color": "green", "width": 2}}
            ))

            self.inspect_fig.data[1].x = self.data[self.anomaly_data['unexplained']].index
            self.inspect_fig.data[1].y = \
                self.data[self.anomaly_data['unexplained']][self.target_column]

    def make_forecast(self, future_periods=720, interval=1, daily=True, weekly=True, yearly=False,
                      seasonality='additive'):
        """

        :param future_periods:
        :param interval:
        :param daily:
        :param weekly:
        :param yearly:
        :param seasonality:
        :return:
        """
        interval = str(interval) + 'h'

        time_forecasting = int(len(self.active_event_names_inspection) * 0.75 + 0.5)
        print(f'>>> Estimated time: {time_forecasting} min')
        progress = widgets.IntProgress(value=0, min=0, max=6,
                                       description='Updating model:', bar_style='info')
        display(progress)
        progress.value += 1

        start = self.data.index[0]
        end = self.data.index[-1] + timedelta(days=30)

        # check if any events are selected
        if self.active_event_names_inspection:
            event_table = self.all_event_data[
                self.all_event_data['eventtype'].apply(lambda x:
                                                       x in self.active_event_names_inspection)]
            event_table.drop(['eventtype'], axis=1, inplace=True)
            mask = (event_table['begin'] > start) & (event_table['end'] <= end)
            filtered_events = event_table.loc[mask]
            converted_holidays = event_table[['begin', 'event']].copy()
            converted_holidays.columns = ["ds", "holiday"]
            forecasting_input = create_model_input(self.data, self.target_column, filtered_events)
            progress.value += 1
            self.filtered_events = filtered_events
            self.converted_holidays = converted_holidays
            self.forecasting_input = forecasting_input
        else:
            filtered_events = None
            self.filtered_events = []
            self.converted_holidays = []
            forecasting_input = create_model_input(self.data, self.target_column, None)
            self.forecasting_input = forecasting_input

        progress.value += 1

        model = Prophet(
            growth='linear',
            changepoint_prior_scale=0.01,
            changepoint_range=0.80,
            interval_width=0.95,
            seasonality_mode=seasonality,
            seasonality_prior_scale=1,
            daily_seasonality=daily,
            weekly_seasonality=weekly,
            yearly_seasonality=yearly
        )

        if filtered_events is not None:
            if not filtered_events.empty:
                for entry in filtered_events['event'].unique().tolist():
                    model.add_regressor('is_' + entry)

        progress.value += 1
        model.fit(forecasting_input)
        progress.value += 1
        future = model.make_future_dataframe(periods=future_periods, freq=interval)
        progress.value += 1

        if filtered_events is not None:
            if not filtered_events.empty:
                for entry in filtered_events['event'].unique().tolist():
                    column_name = 'is_' + entry
                    filtered_event_table = event_table.loc[event_table['event'] == entry]
                    future[column_name] = future['ds'].apply(is_event, args=(filtered_event_table,))

        forecast = model.predict(future)

        progress.value += 1

        self.model = model
        self.forecast = forecast

    def plot_prediction_with_anomalies_and_events(self):
        active_event_data = self.all_event_data[
            self.all_event_data['eventtype'].isin(self.active_event_names_inspection)]

        forecast = self.forecast
        forecast.rename(columns={'ds': 'Time'}, inplace=True)
        forecast = forecast.set_index('Time')

        # calculate possible future anomalies
        forecast_anomaly_detector = AnomalyDetector().with_data(forecast, 'yhat')
        forecast_anomaly_detector.apply_iqr(iq_range=1.5)
        forecast_anomalies = forecast_anomaly_detector.get_anomalies()
        forecast_anomalies.index = forecast_anomalies.index.tz_localize(None)

        forecast_anomaly_data = pd.DataFrame()
        forecast_anomaly_data['anomalies'] = forecast_anomalies

        # ignore y_hat anomalies of the past
        forecast_anomaly_data.loc[forecast_anomaly_data.index <= self.data.index[-1],
                                  'anomalies'] = False

        # ensure dates are actually in datetime format
        active_event_data['begin'] = pd.to_datetime(active_event_data['begin'])
        active_event_data['end'] = pd.to_datetime(active_event_data['end'])

        # Discard rows where 'begin' date is later than 'end' date
        valid_rows = active_event_data[active_event_data['begin'] <= active_event_data['end']]

        # create interval index
        event_intervals = pd.IntervalIndex.from_arrays(valid_rows['begin'], valid_rows['end'],
                                                       closed='both')

        index_in_event_interval = forecast_anomalies.index.map(lambda x:
                                                               any(event_intervals.contains(x)))

        # set explained to True if in interval
        forecast_anomaly_data['explained'] = forecast_anomaly_data['anomalies'] \
                                             & index_in_event_interval

        # set unexplained to True if not in interval
        forecast_anomaly_data['unexplained'] = forecast_anomaly_data['anomalies'] \
                                               & ~forecast_anomaly_data['explained']

        filtered_active_events = crop_data_to_timespan(active_event_data,
                                                       forecast.index[0], forecast.index[-1])

        future_data = self.forecast[self.forecast['Time'] > self.data.index[-1]]

        future_anomaly_detector = AnomalyDetector().with_data(future_data, 'yhat')
        future_anomaly_detector.apply_iqr(iq_range=1.5)
        future_anomalies = future_anomaly_detector.get_anomalies()
        future_anomalies.index = future_anomalies.index.tz_localize(None)

        future_anomaly_data = pd.DataFrame()
        future_anomaly_data['anomalies'] = future_anomalies

        self.predict_fig = go.Figure()

        self.predict_fig.add_trace(
            go.Scatter(
                name=f'Expected {self.target_column}',
                x=list(future_data['Time']), y=list(future_data.yhat),
                line={"color": "rgba(0, 0, 255, 1)", "width": 2}  # Red with 50% opacity
            )
        )

        self.predict_fig.add_trace(
            go.Scatter(
                name=f'Past {self.target_column}',
                x=list(self.data.index), y=list(self.data[self.target_column]),
                line={"color": "rgba(136, 136, 136, 1)", "width": 2}  # Blue with 50% opacity
            )
        )

        self.predict_fig.add_trace(
            go.Scatter(
                name='Confidence Interval',
                x=list(future_data['Time']) + list(future_data['Time'])[::-1],
                y=list(future_data.yhat_upper) + list(future_data.yhat_lower)[::-1],
                fill='toself',
                fillcolor='rgba(76,76,76,0.6)',
                line={"color": "rgba(255, 255, 255, 0)"},
                hoverinfo="skip",
                showlegend=False
            )
        )

        for filtered_event in filtered_active_events.reset_index().values:
            self.predict_fig.add_vrect(x0=filtered_event[2], x1=filtered_event[3],
                                       annotation={"text": filtered_event[1], "textangle": -90},
                                       fillcolor=self.event_color_dict[filtered_event[4]],
                                       opacity=0.3, layer="below", line_width=0)

        self.predict_fig.add_trace(go.Scatter(
            name="Expected Anomalies",
            x=forecast[forecast_anomaly_data['anomalies']].index,
            y=forecast[forecast_anomaly_data['anomalies']]['yhat'],
            mode='markers',
            marker={"color": 'purple', "size": 3, "line": {"color": "Violet", "width": 3}}
        ))

        self.predict_fig.add_trace(go.Scatter(
            name="Past Anomalies",
            x=self.data[self.anomaly_data['anomalies']].index,
            y=self.data[self.anomaly_data['anomalies']][self.target_column],
            mode='markers',
            marker={"color": 'green', "size": 2, "line": {"color": "rgba(34,34,34,1)", "width": 2}}
        ))

        title = 'Time series prediction: ' + self.target_column
        self.predict_fig.update_layout(title_text=title, xaxis=self.plotter.get_x_axis_plot(),
                                       yaxis_visible=False, yaxis_showticklabels=False,
                                       height=self.plot_height)

        with self.predict_output:
            clear_output(wait=True)
            self.display_predict_fig()

    def set_checkboxes_state(self, checkboxes, state):
        for checkbox in checkboxes:
            checkbox.disabled = state

    def clear_vrects(self, fig):
        fig.layout.shapes = ()
        fig.layout.annotations = ()

    def explain_event_impact(self):
        """
        progressbar with explained and unexplained anomalies
        """
        if self.active_event_names_inspection and 'explained' in self.anomaly_data:
            explained = self.anomaly_data[self.anomaly_data['explained']]
            unexplained = self.anomaly_data[self.anomaly_data['unexplained']]
            anomaly_bar = go.Figure(go.Bar(
                x=[len(explained)],
                y=['Anomalies'],
                name='Explained',
                orientation='h',
                marker={"color": 'rgba(0, 255, 50, 0.6)',
                        "line": {"color": "rgba(0, 255, 100, 1.0)", "width": 3}}
            ))
            anomaly_bar.add_trace(go.Bar(
                x=[len(unexplained)],
                y=['Anomalies'],
                name='Unexplained',
                orientation='h',
                marker={"color": 'rgba(255, 0, 50, 0.6)',
                        "line": {"color": "rgba(255, 0, 100, 1.0)", "width": 3}}
            ))
            anomaly_bar.update_layout(barmode='stack', height=230,
                                      title=f'{len(explained)} of '
                                            f'{len(explained) + len(unexplained)} '
                                            f'Anomalies are explained by Events',
                                      title_x=0.5, yaxis_visible=False)
            # Filter forecast only to event regressors
            filtered_forecast = pd.DataFrame()
            # drop upper and lower confidence columns
            if self.forecast is None:
                self.make_forecast()
            working_copy = self.forecast.drop(
                list(self.forecast.filter(regex='_upper|_lower|additive|multiplicative|Time')),
                axis=1)
            # add "is_" to every key to match with forecast column names
            event_category_map_with_prefix = {k: ['is_' + i for i in v]
                                              for k, v in self.event_category_map.items()}
            for new_col, old_cols in event_category_map_with_prefix.items():
                old_cols = [col for col in old_cols if col in working_copy]
                if old_cols:
                    filtered_forecast[new_col] = working_copy[old_cols].sum(axis=1)
            # Identify columns not in dict map
            old_cols = [col for sublist in event_category_map_with_prefix.values()
                        for col in sublist]
            cols_not_in_dict = [col for col in working_copy.columns if col not in old_cols]
            # Copy these columns to filtered_forecast
            filtered_forecast[cols_not_in_dict] = working_copy[cols_not_in_dict]
            try:
                filtered_forecast = filtered_forecast.set_index("ds")
            except KeyError:
                pass
            event_shares = filtered_forecast.drop(list(filtered_forecast.filter(
                regex='trend|daily|weekly|yhat')), axis=1)
            event_counts = event_shares.apply(event_counter)
            # multiply coefficients of events with weakened occurrence
            event_shares_abs = event_shares.abs().max() * round((event_counts + 0.000001) / 2)
            event_shares_rel = event_shares_abs / event_shares_abs.sum()
            # add "(-)" as suffix if impact is negative
            new_index = []
            for event_name in event_shares_rel.index:
                original_val = event_shares.loc[:, event_name].min()
                if original_val < 0:
                    new_index.append(event_name + " (-)")
                else:
                    new_index.append(event_name)
            event_shares_rel.index = new_index
            event_pie = px.pie(values=event_shares_rel, names=event_shares_rel.index,
                               title=f'Share of each event in the explanation '
                                     f'of the {len(explained)} anomalies.')
            event_pie.update_layout(
                legend={
                    "orientation": "v", "yanchor": "bottom", "y": 0.5,
                    "xanchor": "right", "x": 1
                },
                title={"x": 0.5, "xanchor": 'center'}
            )
            out1 = widgets.Output()
            out2 = widgets.Output()
            with out1:
                display(HTML(anomaly_bar.to_html(full_html=False)))
            with out2:
                display(HTML(event_pie.to_html(full_html=False)))
            display(widgets.VBox([out1, out2]))
        else:
            print('No events selected.')


def collect_events_from_path(directory: str, link_collection: dict = None):
    csv_files = Path(directory).glob('*.csv')
    dataframes = []

    for csv_file in csv_files:
        csv_dataframe = pd.read_csv(csv_file, index_col=False)
        old_column_names = csv_dataframe.columns.tolist()
        csv_dataframe.rename(columns={old_column_names[0]: 'event', old_column_names[1]: 'begin',
                           old_column_names[2]: 'end'}, inplace=True)
        csv_dataframe['begin'] = pd.to_datetime(csv_dataframe['begin']).dt.tz_localize(None)
        csv_dataframe['end'] = pd.to_datetime(csv_dataframe['end']).dt.tz_localize(None)
        csv_dataframe['eventtype'] = csv_file.parts[1][:-4].replace("_", " ").title()
        dataframes.append(csv_dataframe)

    if link_collection:
        for event_type, link in link_collection.items():
            cal = Calendar(urlopen(link).read().decode("utf-8"))
            event_array = []
            for event in cal.events:
                event_array.append([event.name, event.begin.datetime,
                                    event.end.datetime, event_type])
            event_dataframe = pd.DataFrame(data=event_array)
            event_dataframe.rename(columns={0: 'event', 1: 'begin', 2: 'end', 3: 'eventtype'},
                                   inplace=True)
            event_dataframe['begin'] = pd.to_datetime(event_dataframe['begin']).dt.tz_localize(None)
            event_dataframe['end'] = pd.to_datetime(event_dataframe['end']).dt.tz_localize(None)
            dataframes.append(event_dataframe)

    event_data = pd.concat(dataframes, ignore_index=True)

    return event_data


def get_event_color_dict(events):
    """
    :param events: Dataframe of events
    :return: dictionary with event type as key and color as value
    """
    color_dict = {}
    v_rect_colors = all_colors_except_red()
    holiday_names = get_holiday_names_list()
    for event in events.reset_index().values:
        if event[4] not in color_dict:
            if event[4] in holiday_names:
                color_dict[event[4]] = 'red'
            else:
                color_dict[event[4]] = v_rect_colors.pop(0)

    return color_dict


def all_colors_except_red():
    css_colors = mcolors.CSS4_COLORS

    def filter_colors(colors_dict):
        excluded_keywords = ["blue", "black"]
        filtered_colors = {}
        for color_name, color_code in colors_dict.items():
            if not any(keyword in color_name.lower() for keyword in excluded_keywords):
                filtered_colors[color_name] = color_code
        return filtered_colors

    filtered_colors_dict = filter_colors(css_colors)
    colors = list(filtered_colors_dict.keys())
    if 'red' in colors:
        colors.remove('red')

    return colors


def get_holiday_names_list():
    with open('holiday_names.json', 'r') as file:
        holiday_names = json.load(file)

    return holiday_names


def make_checkbox_grid(checkboxes: List[widgets.Checkbox], columns: int = 3):
    checkbox_groups = [checkboxes[i::columns] for i in range(columns)]
    grid = widgets.GridBox(
        [checkbox for group in checkbox_groups for checkbox in group],
        layout=widgets.Layout(grid_template_columns=f"repeat({columns}, 1fr)")
    )

    return grid


def crop_data_to_timespan(data: pd.DataFrame, start, end):
    mask = (data['begin'] > start) & (data['end'] <= end)
    cropped_data = data.loc[mask]

    return cropped_data


def event_counter(col):
    col = pd.Series(np.where(col != 0, 1, 0))

    return ((col == 1) & (col.shift(fill_value=0) == 0)).sum()
