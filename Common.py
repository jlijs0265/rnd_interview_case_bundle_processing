import datetime


KST = datetime.timezone(datetime.timedelta(hours=9))
_date_str_format = '%Y-%m-%d'
_time_str_format = '%H:%M:%S'
_datetime_str_format = '{}T{}+09:00'.format(_date_str_format, _time_str_format)


def format_timestamp(_timestamp, _format, tz=None):
    if tz is None:
        tz = KST
    return datetime.datetime.fromtimestamp(_timestamp, tz=tz).strftime(_format)


def timestamp_to_datetime(_timestamp, tz=None):
    if tz is None:
        tz = KST
    return datetime.datetime.fromtimestamp(_timestamp, tz=tz)


def timestamp_to_date_str(_timestamp):
    return format_timestamp(_timestamp, _date_str_format)


def timestamp_to_time_str(_timestamp):
    return format_timestamp(_timestamp, _time_str_format)


def timestamp_to_datetime_str(_timestamp):
    return format_timestamp(_timestamp, _datetime_str_format)


def date_str_to_timestamp(_date_str):
    return datetime.datetime.strptime(_date_str, _date_str_format).timestamp()


def datetime_str_to_timestamp(_datetime_str, datetime_str_format=None):
    if datetime_str_format is None:
        datetime_str_format = _datetime_str_format
    return datetime.datetime.strptime(_datetime_str, datetime_str_format).timestamp()


def midnight_timestamp(_timestamp):
    return date_str_to_timestamp(timestamp_to_date_str(_timestamp))
