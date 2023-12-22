import datetime
import time

class Helpers:
    # Input format: YYYY-MM-DD
    @staticmethod
    def to_unix_time(time_str: str):
        return str(int(time.mktime(datetime.datetime.strptime(time_str, '%Y-%m-%d').timetuple()))) + '000'

    @staticmethod
    def drop_last_line(str_to_cut: str):
        position = str_to_cut.rfind('\n')
        if position != -1:
            return str_to_cut[:position]
        return ''

    @staticmethod
    def df_cols_as_str(df):
        return ',\n    '.join(df.columns.to_list())


class DateRangeGenerator:
    def __init__(self):
        self.__start = datetime.date.today()
        self.__end = self.__start

    def for_past(self, days=0, weeks=0):
        self.__start = self.__start - datetime.timedelta(days=days, weeks=weeks)
        return self

    def generate(self):
        return (self.__start.strftime("%Y-%m-%d"), self.__end.strftime("%Y-%m-%d"))