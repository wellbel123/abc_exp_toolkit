from scipy import stats
import numpy as np
import pandas as pd
import datetime as dt
from typing import List, Tuple, Optional

class FunnelBuilder:
    """
    Build step-by-step funnels with optional per-step time ordering and group splits.

    Parameters
    ----------
    df : pd.DataFrame
        Source events table.
    field_id : str
        Column with user identifier.
    field_event : str
        Column with event name.
    field_time : Optional[str]
        Optional timestamp column. If provided, steps require strictly increasing times per user.
    field_groupby : Optional[str]
        Column to split funnels by groups. If None, a single pseudo-group 'ALL' is used.
    relative, absolute : bool
        Default flags for computing % metrics; can be overridden in build()

    """ 

    def __init__(
        self,
        df: pd.DataFrame,
        field_id: str,
        field_event: str,
        field_time: Optional[str] = None,
        field_groupby: Optional[str] = "group",
        relative: bool = False,
        absolute: bool = False,
    ) -> None:
        self.df = df.copy()
        self.field_id = field_id
        self.field_event = field_event
        self.field_time = field_time
        self.steps: List[List[str]] = []
        self.field_groupby = field_groupby
        self.relative = relative
        self.absolute = absolute

        if field_groupby:
            self.groups = list(pd.Index(self.df[field_groupby].unique()).dropna())
        else:
            self.field_groupby = None
            self.groups = ["ALL"]

    def step(self, *event_names: str) -> "FunnelBuilder":
        """Append a funnel step. Supports multiple events per step."""
        self.steps.append(list(event_names))
        return self

    def _prepare_group_df(self, group: str) -> pd.DataFrame:
        if self.field_groupby is None:
            df_group = self.df
        else:
            df_group = self.df[self.df[self.field_groupby] == group]
        if self.field_time:
            df_group = df_group.copy()
            df_group[self.field_time] = pd.to_datetime(df_group[self.field_time])
            df_group = df_group.sort_values([self.field_id, self.field_time])
        return df_group

    def build(self, relative: Optional[bool] = None, absolute: Optional[bool] = None) -> pd.DataFrame:
        # Final flags: method args override constructor defaults
        rel_flag = self.relative if relative is None else relative
        abs_flag = self.absolute if absolute is None else absolute

        results = []  # list of tuples: (step_label, unique_users_count, group_name)

        # Iterate groups and steps
        for group in self.groups:
            df_group = self._prepare_group_df(group if self.field_groupby else "ALL")

            current_users = None  # set of users who passed the previous step

            for idx, step in enumerate(self.steps):
                step_events = step if isinstance(step, list) else [step]

                step_df = df_group[df_group[self.field_event].isin(step_events)].copy()

                # Keep only users who passed previous step
                if current_users is not None:
                    step_df = step_df[step_df[self.field_id].isin(current_users)].copy()

                # Enforce time strictly increasing with respect to the previous step
                if self.field_time and idx > 0:
                    prev_step = self.steps[idx - 1]
                    prev_events = prev_step if isinstance(prev_step, list) else [prev_step]
                    prev_df = df_group[df_group[self.field_event].isin(prev_events)].copy()

                    # earliest time of previous step per user
                    prev_times = prev_df.groupby(self.field_id, as_index=True)[self.field_time].min()

                    step_df = step_df.merge(
                        prev_times.rename("__prev_ts__"),
                        left_on=self.field_id,
                        right_index=True,
                        how="inner",
                    )
                    step_df = step_df[step_df[self.field_time] >= step_df["__prev_ts__"]].copy()

                users = pd.Index(step_df[self.field_id].unique())
                results.append((
                    ", ".join(step) if isinstance(step, list) else step,
                    int(users.size),
                    group if self.field_groupby else "ALL",
                ))

                # Update survivors for the next step
                current_users = set(users)

        # Build tidy result
        df_result = pd.DataFrame(results, columns=["event", "unique_users", "group_name"]) \
                         .astype({"unique_users": int, "group_name": str})

        # Wide format: one column per group; keep step order
        df_wide = df_result.pivot(index="event", columns="group_name", values="unique_users").reset_index()
        # Ensure rows are ordered by declared steps
        step_labels = [", ".join(s) if isinstance(s, list) else s for s in self.steps]
        df_wide = df_wide.set_index("event").loc[step_labels].reset_index()

        # Compute % metrics on top of wide counts, avoiding SettingWithCopyWarning
        group_cols = [c for c in df_wide.columns if c != "event"]

        if rel_flag:
            for g in group_cols:
                col_name = f"{g}_rel_%"
                prev = None
                rel_vals = []
                for val in df_wide[g].tolist():
                    if prev is None:
                        rel_vals.append(100.0)
                    else:
                        rel_vals.append(round((val / prev) * 100, 2) if prev > 0 else 0.0)
                    prev = val
                df_wide.loc[:, col_name] = rel_vals

        if abs_flag:
            for g in group_cols:
                col_name = f"{g}_abs_%"
                first = float(df_wide[g].iloc[0])
                abs_vals = [round((v / first) * 100, 2) if first > 0 else 0.0 for v in df_wide[g].tolist()]
                df_wide.loc[:, col_name] = abs_vals

        return df_wide



def compute_step_stats(
    funnel_df: pd.DataFrame,
    groups: List[str],
    hypothesis: str = "two-sided",   # 'two-sided' | 'one-sided' (tests if B > A)
    relative: bool = True,           # conversions vs previous step
    absolute: bool = True,           # conversions vs first step
    delta0: float = 0.0,             # margin in proportions (e.g., 0.05 = 5 pp) - for non-inferiority tests 
    alpha: float = 0.05,             # significance level
    correction: str = "none",        # 'none' | 'bonferroni'
    m_tests: int = 1,                # number of hypotheses
) -> pd.DataFrame:
    """
    For each funnel step (row in `funnel_df`) compute:
      - relative conversions (% of previous step) for A and B
      - absolute conversions (% of first step) for A and B
      - differences in percentage points (B - A) for rel/abs
      - z and p-values (two-proportion z-test) for rel/abs with margin `delta0`
      - absolute difference in counts on the step (B - A)
      - Bonferroni-adjusted alpha (if requested) and significance flags

    Requirements:
      - `funnel_df` must be wide: columns include 'event', group A and group B names.
      - Row order equals funnel step order.
    """
    # input checks 
    if "event" not in funnel_df.columns:
        raise ValueError("`funnel_df` must contain column 'event'.")
    if len(groups) != 2:
        raise ValueError("`groups` must be a list of two names, e.g. ['A','B'].")
    g1, g2 = groups
    missing = [g for g in (g1, g2) if g not in funnel_df.columns]
    if missing:
        raise ValueError(f"Missing columns in `funnel_df`: {missing}. Present: {list(funnel_df.columns)}")
    if hypothesis not in ("two-sided", "one-sided"):
        raise ValueError("`hypothesis` must be 'two-sided' or 'one-sided'.")
    if correction not in ("none", "bonferroni"):
        raise ValueError("`correction` must be 'none' or 'bonferroni'.")
    if m_tests is None or m_tests < 1:
        raise ValueError("`m_tests` must be a positive integer (>=1).")

    # Bonferroni-adjusted alpha 
    alpha_adj = alpha / m_tests if correction == "bonferroni" else alpha


    def z_test(n1: float, n2: float, x1: float, x2: float) -> Tuple[float, float]:
        """
        Two-proportion z-test with margin `delta0`.
        Tests H0: p2 - p1 <= delta0  vs  H1: p2 - p1 > delta0  (one-sided)
        or H0: p2 - p1 =  delta0     vs  H1: p2 - p1 != delta0 (two-sided)
        Returns (z, p).
        """
        if n1 <= 0 or n2 <= 0:
            return (np.nan, np.nan)
        p1, p2 = x1 / n1, x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se == 0:
            return (np.nan, np.nan)
        z = ((p2 - p1) - delta0) / se
        if hypothesis == "two-sided":
            p = 2 * stats.norm.sf(abs(z))
        else:  # one-sided (right tail: B > A by at least delta0)
            p = stats.norm.sf(z)
        return (z, p)

    def conv_block(prefix: str, n1: float, n2: float, x1: float, x2: float) -> dict:
        """
        Compute conversions (%), diff in pp, and z/p with the given denominators.
        `prefix` ->  {'rel', 'abs'}.
        """
        out = {}
        conv1 = (x1 / n1 * 100.0) if n1 > 0 else 0.0
        conv2 = (x2 / n2 * 100.0) if n2 > 0 else 0.0
        diff_pp = conv2 - conv1
        if (n1 > 0) and (n2 > 0) and (x1 > 0) and (x2 > 0):
            z, p = z_test(n1, n2, x1, x2)
        else: 
            z,p=np.nan, np.nan 

        out[f"{g1}_{prefix}_%"] = round(conv1, 4)
        out[f"{g2}_{prefix}_%"] = round(conv2, 4)
        out[f"diff_{prefix}_pp"] = round(diff_pp, 4)
        out[f"z_{prefix}"] = z
        out[f"p_value_{prefix}"] = p
        out[f"sig_{prefix}"] = (p < alpha_adj) if not np.isnan(p) else False
        return out

    # prep first/prev denominators 
    x1_first = float(funnel_df[g1].iloc[0])
    x2_first = float(funnel_df[g2].iloc[0])
    prev1 = prev2 = None

    rows = []
    for i, r in funnel_df.iterrows():
        event = r["event"]
        x1 = float(r[g1])
        x2 = float(r[g2])

        out = {
            "event": event,
            g1: x1,
            g2: x2,
            "diff_abs": x2 - x1,   # raw difference in counts at this step
        }

        if i == 0:
            # First step: 100% by definition
            if relative:
                out[f"{g1}_rel_%"] = 100.0 if x1 > 0 else 0.0
                out[f"{g2}_rel_%"] = 100.0 if x2 > 0 else 0.0
                out["diff_rel_pp"] = out[f"{g2}_rel_%"] - out[f"{g1}_rel_%"]
                out["z_rel"] = np.nan
                out["p_value_rel"] = np.nan
                out["sig_rel"] = False
            if absolute:
                out[f"{g1}_abs_%"] = 100.0 if x1_first > 0 else 0.0
                out[f"{g2}_abs_%"] = 100.0 if x2_first > 0 else 0.0
                out["diff_abs_pp"] = out[f"{g2}_abs_%"] - out[f"{g1}_abs_%"]
                out["z_abs"] = np.nan
                out["p_value_abs"] = np.nan
                out["sig_abs"] = False
            rows.append(out)
            prev1, prev2 = x1, x2
            continue

        if relative:
            out.update(conv_block("rel", n1=prev1, n2=prev2, x1=x1, x2=x2))
        if absolute:
            out.update(conv_block("abs", n1=x1_first, n2=x2_first, x1=x1, x2=x2))

        rows.append(out)
        prev1, prev2 = x1, x2

    return pd.DataFrame(rows)


def generate_synthetic_funnel(
    n_per_group: int = 30000,
    groups = ("A", "B"),
    # Steps: a list of event lists (multievents per step)
    steps = (["view", "view1"], ["signup"], ["buy"]),
    # step-by-step conversions for each group (from previous steps)
    conv = {
        "A": (1.00, 0.35, 0.12),  #  view→signup ~35%, signup→buy ~12%
        "B": (1.00, 0.38, 0.14),  #  B is better than A 
    },
    start_date: str = "2024-01-01",
    max_days_between_steps = (1, 3, 7),  # max. delays before step i (in days)
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns pd.DataFrame with columns: user_id, event, ts, group
    - steps: a list of steps, each step can have several events (multi-events)
    - conv[g][i] — probability of passing step i (conditional to the previous step)
    - ts -  monotonously increase in steps
    """
    # creates a random number generator object (NumPy’s random API) 
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.to_datetime(start_date)

    for g in groups:
        if g not in conv:
            raise ValueError(f"Missing conv for group {g}")
        probs = conv[g]
        if len(probs) != len(steps):
            raise ValueError("len(conv[group]) must match len(steps)")

        # create user_ids
        user_ids = np.array([f"{g}_{i:06d}" for i in range(1, n_per_group + 1)])

        # An array of masks of users who have reached the current step, all Trues 
        alive_mask = np.ones(n_per_group, dtype=bool)

        # Time base of the first event (starting noise)
        # we give each user a random "day 0"
        base_offsets = rng.integers(0, max_days_between_steps[0] if len(max_days_between_steps) > 0 else 1, size=n_per_group)
        last_ts = base + pd.to_timedelta(base_offsets, unit="D")

        for step_idx, event_list in enumerate(steps):
            event_list = list(event_list) 
            p = probs[step_idx]
            # all Falses until reaches the steps 
            step_take = np.zeros(n_per_group, dtype=bool)
            step_take[alive_mask] = rng.random(alive_mask.sum()) < p

            # users with this step 
            idx = np.where(step_take)[0]
            if idx.size == 0:
                # no one has passed this step in this group.
                alive_mask[:] = False
                continue

            # assign a specific event to the user at this step
            # chosen_events is an array of event indexes for each user.
            chosen_events = rng.integers(0, len(event_list), size=idx.size)
            events = [event_list[k] for k in chosen_events]

            # time for a step: always later than the previous one
            # adding a random delay within max_days_between_steps[step_idx]
            delay_days = max_days_between_steps[step_idx] if step_idx < len(max_days_between_steps) else 1
            delays = rng.integers(0, max(1, delay_days), size=idx.size)
            step_ts = last_ts[idx] + pd.to_timedelta(delays, unit="D")

            # collect tuples in a general list
            rows.extend(
                (user_ids[i], events[j], step_ts[j], g)
                for j, i in enumerate(idx)
            )

            alive_mask = step_take
            # upload last_ts only for them who passed 
            last_ts = (base + pd.to_timedelta(base_offsets, unit="D")).to_numpy()


    df = pd.DataFrame(rows, columns=["user_id", "event", "ts", "group"])
    df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)
    return df