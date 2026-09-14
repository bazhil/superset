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

import pandas as pd

from superset.charts.pivot_aggregators import (
    first_along_axis,
    first_value,
    last_along_axis,
    last_value,
    list_unique_values,
    sample_standard_deviation,
    sample_variance,
    sample_variance_along_axis,
)


def test_first_and_last_skip_nulls() -> None:
    series = pd.Series([None, "alpha", "beta", None])
    assert first_value(series) == "alpha"
    assert last_value(series) == "beta"
    assert first_value(pd.Series([None, None])) is None


def test_sample_variance_and_std_on_generated_values() -> None:
    series = pd.Series([2, 4, 4, 4, 5, 5, 7, 9])
    assert sample_variance(series) == float(series.var())
    assert sample_standard_deviation(series) == float(series.std())
    assert sample_variance(pd.Series([3])) == 0
    assert sample_standard_deviation(pd.Series(["x", "y"])) == 0


def test_list_unique_values_is_stable_and_collapses_numeric_aliases() -> None:
    series = pd.Series([1, "1.0", 1.0, "west", "west"])
    listed = list_unique_values(series)
    tokens = [part.strip() for part in listed.split(",")]
    assert tokens[0] == "1"
    assert tokens.count("west") == 1
    # a second 1 / "1.0" must not appear as its own token
    assert tokens.count("1") + tokens.count("1.0") == 1
    assert list_unique_values(pd.Series([True, False, True])) == "True, False"


def test_reducers_accept_dataframe_axis_for_subtotals() -> None:
    frame = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    first_row = first_along_axis(frame, axis=1)
    last_col = last_along_axis(frame, axis=0)
    assert list(first_row) == [10, 20]
    assert list(last_col) == [20, 40]
    variances = sample_variance_along_axis(frame, axis=0)
    assert variances["a"] == float(pd.Series([10, 20]).var())


def test_first_last_on_generated_pivot_source() -> None:
    """First/Last return scalars from source rows, not Series slices."""
    source = pd.DataFrame(
        {
            "region": ["EMEA", "EMEA", "APAC", "APAC"],
            "metric": [10, 30, 5, 7],
        }
    )
    pivoted = source.pivot_table(
        index="region",
        values="metric",
        aggfunc=first_along_axis,
    )
    assert pivoted.loc["EMEA", "metric"] == 10
    pivoted_last = source.pivot_table(
        index="region",
        values="metric",
        aggfunc=last_along_axis,
    )
    assert pivoted_last.loc["EMEA", "metric"] == 30
