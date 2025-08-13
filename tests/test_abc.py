# import sys
# sys.path.append('..')

from src.abc_exp_toolkit import FunnelBuilder, compute_step_stats
import pandas as pd
import numpy as np
from scipy import stats
import pytest
import math

@pytest.fixture()
def small_events_df() -> pd.DataFrame:
    # Synthetic, fully deterministic mini log with time ordering
    rows = [
        # group A users
        ("A_u1", "view",   "2024-01-01", "A"),
        ("A_u1", "signup", "2024-01-02", "A"),
        ("A_u1", "buy",    "2024-01-03", "A"),
        ("A_u2", "view",   "2024-01-01", "A"),
        ("A_u2", "signup", "2024-01-03", "A"),
        # A_u2 stops at signup
        ("A_u3", "view1",  "2024-01-01", "A"),  # multi-event first step
        # A_u3 churns after view1

        # group B users
        ("B_u1", "view",   "2024-01-01", "B"),
        ("B_u1", "signup", "2024-01-01", "B"),  # same day allowed
        ("B_u1", "buy",    "2024-01-05", "B"),
        ("B_u2", "view1",  "2024-01-02", "B"),
        ("B_u2", "signup", "2024-01-04", "B"),
        # B_u2 churns before buy
        ("B_u3", "view",   "2024-01-01", "B"),  
    ]
    df = pd.DataFrame(rows, columns=["user_id", "event", "ts", "group"]) 
    return df


def test_funnelbuilder_basic_counts_and_multievent(small_events_df: pd.DataFrame):
    fb = (
        FunnelBuilder(
            df=small_events_df,
            field_id="user_id",
            field_event="event",
            field_time="ts",
            field_groupby="group",
        )
        .step("view", "view1")
        .step("signup")
        .step("buy")
    )

    out = fb.build()

    # Expected unique user counts per group per step
    # Step 1 (view/view1): A={u1,u2,u3}=3, B={u1,u2,u3}=3
    # Step 2 (signup):     A={u1,u2}=2,   B={u1,u2}=2
    # Step 3 (buy):        A={u1}=1,      B={u1}=1
    assert list(out["event"]) == ["view, view1", "signup", "buy"]
    assert out.loc[out["event"] == "view, view1", "A"].iloc[0] == 3
    assert out.loc[out["event"] == "view, view1", "B"].iloc[0] == 3
    assert out.loc[out["event"] == "signup", "A"].iloc[0] == 2
    assert out.loc[out["event"] == "signup", "B"].iloc[0] == 2
    assert out.loc[out["event"] == "buy", "A"].iloc[0] == 1
    assert out.loc[out["event"] == "buy", "B"].iloc[0] == 1

def test_funnelbuilder_relative_and_absolute(small_events_df: pd.DataFrame):
    fb = (
        FunnelBuilder(
            df=small_events_df,
            field_id="user_id",
            field_event="event",
            field_time="ts",
            field_groupby="group",
            relative=True,
            absolute=True,
        )
        .step("view", "view1")
        .step("signup")
        .step("buy")
    )
    out = fb.build()

    # Columns for rel/abs should be present
    assert {"A_rel_%", "B_rel_%", "A_abs_%", "B_abs_%"}.issubset(set(out.columns))

    # Validate percentages for A
    # rel: [100, 2/3*100, 1/2*100]; abs: [100, 2/3*100, 1/3*100]
    a_rel = out[["event", "A_rel_%"]].set_index("event").squeeze()
    a_abs = out[["event", "A_abs_%"]].set_index("event").squeeze()
    assert math.isclose(a_rel["view, view1"], 100.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(a_rel["signup"], round(2/3*100, 2), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(a_rel["buy"], round(1/2*100, 2), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(a_abs["view, view1"], 100.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(a_abs["signup"], round(2/3*100, 2), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(a_abs["buy"], round(1/3*100, 2), rel_tol=0, abs_tol=1e-9)


def test_funnelbuilder_time_sequence_filtering():
    # Create a dataset where a user has a later-step timestamp BEFORE the previous step
    rows = [
        ("u1", "view",   "2024-01-10", "A"),
        ("u1", "signup", "2024-01-05",  "A"),  # out-of-order, should be filtered out by time rule
        ("u1", "signup", "2024-01-12",  "A"),  # valid one remains
        ("u2", "view",   "2024-01-01", "A"),
        # u2 has no signup
    ]
    df = pd.DataFrame(rows, columns=["user_id", "event", "ts", "group"]) 

    fb = (
        FunnelBuilder(df, "user_id", "event", field_time="ts", field_groupby="group")
        .step("view")
        .step("signup")
    )
    out = fb.build()

    # Step1: A has {u1, u2} = 2; Step2: only u1 qualifies (the late one on Jan-12)
    assert out.loc[out["event"] == "view", "A"].iloc[0] == 2
    assert out.loc[out["event"] == "signup", "A"].iloc[0] == 1

def test_compute_step_stats_input_validation():
    funnel_df = pd.DataFrame({"event": ["s1"], "A": [10], "B": [12]})

    with pytest.raises(ValueError):
        compute_step_stats(funnel_df=funnel_df.drop(columns=["event"]), groups=["A", "B"])  # no 'event'

    with pytest.raises(ValueError):
        compute_step_stats(funnel_df=funnel_df, groups=["A"])  # not two groups

    with pytest.raises(ValueError):
        compute_step_stats(funnel_df=funnel_df.rename(columns={"A": "X"}), groups=["A", "B"])  # missing column

    with pytest.raises(ValueError):
        compute_step_stats(funnel_df=funnel_df, groups=["A", "B"], hypothesis="weird")

    with pytest.raises(ValueError):
        compute_step_stats(funnel_df=funnel_df, groups=["A", "B"], correction="sidak")

    with pytest.raises(ValueError):
        compute_step_stats(funnel_df=funnel_df, groups=["A", "B"], m_tests=0)

def test_compute_step_stats_zero_denominator_handling():
    # A has no users after first step; B has some
    fdf = pd.DataFrame({
        "event": ["s1", "s2"],
        "A": [10, 0],
        "B": [10, 4],
    })
    res = compute_step_stats(
        funnel_df=fdf,
        groups=["A", "B"],
        relative=True,
        absolute=True,
    )

    r1 = res.iloc[1]
    # Relative conversions: A_rel_% becomes 0.0 due to n1=prev1=0; p-value should be NaN (no denominator)
    assert math.isclose(r1["A_rel_%"], 0.0)
    assert math.isclose(r1["B_rel_%"], 40.0)
    assert math.isnan(r1["p_value_rel"])
