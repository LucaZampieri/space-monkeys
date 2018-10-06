"""
conversions.py

This files contains some helpers functions (we may split it in subfiles when the
number of helper functions increases)
"""


def unix_time_to_mjd(time_in_unix):
    """
    Converts the time format from unix to MJD (Modified Julian Date)
    86400 is the # of sec per 24 hours
    40587 is the unix epoch in mjd
    """
    time_in_mjd = time_in_unix / 86400 + 40587
    return time_in_mjd


def mjd_time_to_unix(time_in_mjd):
    time_in_unix = (time_in_mjd-40587)*86400
    return time_in_unix
