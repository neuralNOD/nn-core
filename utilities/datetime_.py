# -*- encoding: utf-8 -*-

"""
A list of utility functions related to datetime objects. The
module file is named as `datetime_` to keep the name different
from that available under python core packages.
"""

import datetime as dt

def date_range(start : dt.date, end : dt.date) -> dt.date:
    """
    Given a start and an end date, yeilds all dates between
    the time span. TODO add `kwargs` if required on some interval.
    """

    span = end - start # dt object can be subtracted easily
    for i in range(span.days + 1):
        yield start + dt.timedelta(days = i)
