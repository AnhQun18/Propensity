"""Nhóm cố định để so sánh các tháng và cửa sổ theo cùng định nghĩa."""

import numpy as np
import pandas as pd


CORE_FEATURES = ("trx_event_count", "trx_active_days", "trx_amount_mean", "trx_recency_days",
                 "geo_event_count", "geo_unique_locations", "dialog_event_count", "any_active_days")


def feature_group(values, feature):
    if "recency" in feature:
        bins, labels = [-np.inf, 7, 30, 90, np.inf], ["<=7", "(7,30]", "(30,90]", ">90"]
    elif "amount" in feature:
        # Không dùng quantile của tương lai để tạo ranh giới nhóm cho tháng quá khứ.
        bins, labels = [-np.inf, 0, 1, 10, 100, np.inf], ["<=0", "(0,1]", "(1,10]", "(10,100]", ">100"]
    else:
        bins, labels = [-np.inf, 0, 5, 20, 100, np.inf], ["0", "(0,5]", "(5,20]", "(20,100]", ">100"]
    grouped = pd.cut(values, bins=bins, labels=labels).astype("string").fillna("<MISSING>")
    return grouped


def window_features(frame, days):
    return [name+"_"+str(days)+"d" for name in CORE_FEATURES if name+"_"+str(days)+"d" in frame]


def source_combination(frame, suffix=""):
    return (frame["has_trx"+suffix].astype(int).astype(str)
            + frame["has_geo"+suffix].astype(int).astype(str)
            + frame["has_dialog"+suffix].astype(int).astype(str))
