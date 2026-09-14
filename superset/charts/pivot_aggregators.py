# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Leaf reducers for Pivot Table v2 export (CSV, Excel, reports).

These match the aggregators in the pivot plugin: they return a scalar, coerce
numeric inputs where the chart would, and accept both a ``Series`` (leaf cells)
and a ``DataFrame`` plus ``axis`` (row/column totals inserted after the pivot).
"""

from __future__ import annotations

import math
import numbers
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

import pandas as pd

NumericReducer = Callable[..., Any]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _along_axis(reducer: Callable[[pd.Series], Any]) -> NumericReducer:
    """Lift a Series reducer so totals can pass a DataFrame and ``axis``."""

    def wrapped(
        values: pd.Series | pd.DataFrame, axis: int = 0, **kwargs: Any
    ) -> Any:
        if isinstance(values, pd.DataFrame):
            return values.apply(reducer, axis=axis)
        return reducer(values)

    return wrapped


def first_value(series: pd.Series) -> Any:
    """First non-null value, matching the plugin's First aggregator."""
    present = series.dropna()
    return None if present.empty else present.iloc[0]


def last_value(series: pd.Series) -> Any:
    """Last non-null value, matching the plugin's Last aggregator."""
    present = series.dropna()
    return None if present.empty else present.iloc[-1]


def sample_variance(series: pd.Series) -> Any:
    """Sample variance (ddof=1); a single point is defined as 0."""
    numeric = _numeric(series).dropna()
    if len(numeric) <= 1:
        return 0
    return float(numeric.var())


def sample_standard_deviation(series: pd.Series) -> Any:
    """Sample standard deviation (ddof=1); a single point is defined as 0."""
    numeric = _numeric(series).dropna()
    if len(numeric) <= 1:
        return 0
    return float(numeric.std())


def _unique_key(value: Any) -> tuple[str, Any]:
    """
    Deduplicate List Unique Values by numeric meaning when possible.

    ``1``, ``1.0`` and ``"1.00"`` collapse to one entry; booleans stay distinct
    from numbers; unparsable strings keep their original identity.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("missing", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Decimal):
        return ("num", value)
    if isinstance(value, numbers.Number):
        return ("num", Decimal(str(value)))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ("str", stripped)
        try:
            return ("num", Decimal(stripped))
        except (InvalidOperation, ValueError):
            return ("str", value)
    return ("str", str(value))


def list_unique_values(series: pd.Series) -> str:
    """Comma-separated unique values in first-seen order."""
    seen: set[tuple[str, Any]] = set()
    parts: list[str] = []
    for cell in pd.Series.unique(series):
        key = _unique_key(cell)
        if key in seen:
            continue
        seen.add(key)
        parts.append(str(cell))
    return ", ".join(parts)


first_along_axis = _along_axis(first_value)
last_along_axis = _along_axis(last_value)
sample_variance_along_axis = _along_axis(sample_variance)
sample_std_along_axis = _along_axis(sample_standard_deviation)
list_unique_along_axis = _along_axis(list_unique_values)
