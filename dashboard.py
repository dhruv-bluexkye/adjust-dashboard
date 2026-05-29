"""
Adjust Analytics Dashboard — 16Arena
Fetches hourly CSV.GZ dumps from Azure Blob Storage and renders
interactive charts across Acquisition, Retention, Events, Attribution,
Device/Geo, App-Version, and a full Campaign/Network Deep Dive.
"""

import gzip
import io
import csv
import hashlib
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth
from azure.storage.blob import ContainerClient

# ── Config (read from .streamlit/secrets.toml — never hardcoded) ───────────────
ACCOUNT_NAME   = st.secrets["azure"]["account_name"]
CONTAINER_NAME = st.secrets["azure"]["container_name"]
SAS_TOKEN      = st.secrets["azure"]["sas_token"]
CONTAINER_URL  = (
    f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}?{SAS_TOKEN}"
)
CACHE_DIR = Path("blob_cache")
CACHE_DIR.mkdir(exist_ok=True)

PII_COLS = {
    "user_agent", "ip_address", "push_token", "idfa", "gps_adid",
    "idfa_md5", "gps_adid_md5", "idfa_md5_hex", "idfa_upper",
    "idfa||gps_adid", "idfa||android_id", "idfa||gps_adid||fire_adid",
    "idfv||google_app_set_id",
}

# ── Azure helpers ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_client():
    return ContainerClient.from_container_url(CONTAINER_URL)


@st.cache_data(ttl=300, show_spinner=False)
def list_blob_names():
    return [b.name for b in get_client().list_blobs()]


def blob_cache_path(name: str) -> Path:
    h = hashlib.md5(name.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.parquet"


def combined_cache_path(blob_names: tuple) -> Path:
    key = hashlib.md5("|".join(sorted(blob_names)).encode()).hexdigest()[:16]
    return CACHE_DIR / f"combined_{key}.parquet"


def read_blob(client, name: str) -> pd.DataFrame:
    cache = blob_cache_path(name)
    if cache.exists():
        return pd.read_parquet(cache)
    raw = client.get_blob_client(name).download_blob().readall()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    delimiter = "\t" if name.lower().replace(".gz", "").endswith(".tsv") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.columns = [c.strip("{}") for c in df.columns]
    df.to_parquet(cache, index=False)
    return df


@st.cache_data(show_spinner=False)
def load_data(blob_names: tuple) -> pd.DataFrame:
    combined = combined_cache_path(blob_names)
    if combined.exists():
        return pd.read_parquet(combined)

    client = get_client()
    frames = []
    n_cached = sum(1 for n in blob_names if blob_cache_path(n).exists())
    n_new = len(blob_names) - n_cached
    status_text = (
        f"Downloading {n_new} new blob(s), {n_cached} from disk cache…"
        if n_new else f"Loading {n_cached} blob(s) from disk cache…"
    )
    bar = st.progress(0, text=status_text)
    for i, name in enumerate(blob_names):
        try:
            df = read_blob(client, name)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            st.warning(f"Skipped {name}: {exc}")
        bar.progress((i + 1) / len(blob_names))
    bar.empty()
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(combined, index=False)
    return result


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["created_at", "installed_at", "created_at_hour"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mask = df[col] > 1e9
            df[col] = df[col].where(
                ~mask, pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
            )
    for col in ["session_count", "lifetime_session_count", "time_spent", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["is_organic", "is_reattributed", "is_redownload"]:
        if col in df.columns:
            df[col] = df[col].map(
                {"1": True, "0": False, "true": True, "false": False,
                 True: True, False: False}
            )
    # normalise empty strings → NaN in key dimensions
    for col in ["campaign_name", "network_name", "adgroup_name",
                "tracker_name", "creative_name", "country", "city"]:
        if col in df.columns:
            df[col] = df[col].replace("", pd.NA)
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────
CHART_H = 380

def bar(df, x, y, title, color=None, h=CHART_H, orientation="v"):
    fig = px.bar(df, x=x, y=y, color=color, title=title, orientation=orientation,
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=h, margin=dict(t=40, b=0))
    return fig


def hbar(df, x, y, title, h=CHART_H):
    return bar(df, x, y, title, h=h, orientation="h")


def line(df, x, y, title, color=None, h=CHART_H):
    fig = px.line(df, x=x, y=y, color=color, title=title, markers=True,
                  color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=h, margin=dict(t=40, b=0))
    return fig


def pie(df, names, values, title, h=CHART_H):
    fig = px.pie(df, names=names, values=values, title=title, hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=h, margin=dict(t=40, b=0))
    return fig


def kpi(label, value, delta=None):
    st.metric(label, value, delta)


def safe_date(series):
    """Return True if series is datetime-typed."""
    return pd.api.types.is_datetime64_any_dtype(series)


# ── Segment deep-dive helper ───────────────────────────────────────────────────

def render_segment_drill(seg_df: pd.DataFrame, full_df: pd.DataFrame, label: str):
    """
    Given seg_df = all rows for users acquired via the selected segment,
    render the full deep-dive analysis.
    """
    if seg_df.empty:
        st.info(f"No data found for **{label}**.")
        return

    installs_s = seg_df[seg_df["activity_kind"] == "install"] if "activity_kind" in seg_df.columns else pd.DataFrame()
    events_s   = seg_df[seg_df["activity_kind"] == "event"]   if "activity_kind" in seg_df.columns else pd.DataFrame()
    sessions_s = seg_df[seg_df["activity_kind"] == "session"] if "activity_kind" in seg_df.columns else pd.DataFrame()

    unique_users_s = seg_df["adid"].nunique() if "adid" in seg_df.columns else 0
    avg_events = len(events_s) / unique_users_s if unique_users_s else 0
    avg_sessions = len(sessions_s) / unique_users_s if unique_users_s else 0

    # ── KPI strip ────────────────────────────────────────────────────────────
    st.markdown(f"### {label}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi("Users", f"{unique_users_s:,}")
    with c2: kpi("Installs", f"{len(installs_s):,}")
    with c3: kpi("Events", f"{len(events_s):,}")
    with c4: kpi("Sessions", f"{len(sessions_s):,}")
    with c5: kpi("Avg Events/User", f"{avg_events:.1f}")
    with c6: kpi("Avg Sessions/User", f"{avg_sessions:.1f}")

    st.divider()

    # ── Events breakdown ─────────────────────────────────────────────────────
    st.markdown("#### What do these users do? — Events")
    if not events_s.empty and "event_name" in events_s.columns:
        col1, col2 = st.columns([3, 2])
        with col1:
            top_ev = (
                events_s["event_name"].fillna("(blank)")
                .value_counts().head(20).reset_index()
            )
            top_ev.columns = ["event", "count"]
            top_ev["pct_users"] = top_ev["count"] / unique_users_s * 100
            st.plotly_chart(
                hbar(top_ev, "count", "event", f"Events fired by users from {label}", h=420),
                use_container_width=True,
            )
        with col2:
            st.markdown("**Events table**")
            st.dataframe(top_ev.rename(columns={"count": "fires", "pct_users": "per 100 users"}),
                         use_container_width=True, height=420)

        # Events over time
        if "created_at" in events_s.columns and safe_date(events_s["created_at"]):
            top5 = events_s["event_name"].value_counts().head(5).index.tolist()
            ev_trend = (
                events_s[events_s["event_name"].isin(top5)]
                .assign(date=events_s["created_at"].dt.date)
                .groupby(["date", "event_name"]).size().reset_index(name="count")
            )
            st.plotly_chart(
                line(ev_trend, "date", "count", "Top 5 Events — Daily Trend", color="event_name"),
                use_container_width=True,
            )

        # Events per unique user distribution
        if "adid" in events_s.columns:
            epu = events_s.groupby("adid").size().clip(upper=30)
            epu_df = epu.value_counts().sort_index().reset_index()
            epu_df.columns = ["events_fired", "user_count"]
            st.plotly_chart(
                bar(epu_df, "events_fired", "user_count",
                    "Events per User Distribution (capped 30)"),
                use_container_width=True,
            )
    else:
        st.info("No event records for this segment.")

    st.divider()

    # ── Session / Retention signals ───────────────────────────────────────────
    st.markdown("#### Retention signals — Sessions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if "lifetime_session_count" in seg_df.columns:
            lsc = pd.to_numeric(seg_df["lifetime_session_count"], errors="coerce").dropna()
            if len(lsc):
                buckets = pd.cut(lsc, bins=[0, 1, 3, 7, 14, 30, 9999],
                                 labels=["1", "2-3", "4-7", "8-14", "15-30", "30+"])
                lsc_df = buckets.value_counts().sort_index().reset_index()
                lsc_df.columns = ["sessions", "users"]
                st.plotly_chart(
                    bar(lsc_df, "sessions", "users", "Lifetime Session Buckets"),
                    use_container_width=True,
                )
    with col2:
        # Stickiness: 1-session vs returning
        if "adid" in seg_df.columns and "activity_kind" in seg_df.columns:
            all_sess = seg_df[seg_df["activity_kind"] == "session"]
            if not all_sess.empty:
                sess_per_user = all_sess.groupby("adid").size()
                sticky = pd.Series({
                    "1 session (churned)": (sess_per_user == 1).sum(),
                    "2-3 sessions": ((sess_per_user >= 2) & (sess_per_user <= 3)).sum(),
                    "4+ sessions (retained)": (sess_per_user >= 4).sum(),
                })
                sticky_df = sticky.reset_index()
                sticky_df.columns = ["bucket", "users"]
                st.plotly_chart(
                    pie(sticky_df, "bucket", "users", "Stickiness Breakdown"),
                    use_container_width=True,
                )
    with col3:
        if "time_spent" in seg_df.columns:
            ts = pd.to_numeric(seg_df["time_spent"], errors="coerce").dropna()
            if len(ts):
                st.metric("Avg Time Spent", f"{ts.mean():.0f}s")
                st.metric("Median Time Spent", f"{ts.median():.0f}s")
                st.metric("Max Time Spent", f"{ts.max():.0f}s")

    if not sessions_s.empty and "created_at" in sessions_s.columns and safe_date(sessions_s["created_at"]):
        sess_daily = (
            sessions_s.assign(date=sessions_s["created_at"].dt.date)
            .groupby("date").size().reset_index(name="sessions")
        )
        st.plotly_chart(line(sess_daily, "date", "sessions", "Daily Sessions from Segment"),
                        use_container_width=True)

    st.divider()

    # ── Geo & Device breakdown ────────────────────────────────────────────────
    st.markdown("#### Geo & Device — Where are these users from?")
    col1, col2 = st.columns(2)
    with col1:
        if "country" in seg_df.columns:
            ctry = seg_df["country"].fillna("Unknown").value_counts().head(20).reset_index()
            ctry.columns = ["country", "count"]
            fig_map = px.choropleth(
                ctry, locations="country", locationmode="country names",
                color="count", title=f"Users by Country — {label}",
                color_continuous_scale="Blues",
            )
            fig_map.update_layout(height=380, margin=dict(t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
    with col2:
        if "city" in seg_df.columns:
            city = seg_df["city"].fillna("Unknown").value_counts().head(12).reset_index()
            city.columns = ["city", "count"]
            st.plotly_chart(hbar(city, "count", "city", "Top Cities"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "os_name" in seg_df.columns:
            os_df = seg_df["os_name"].fillna("Unknown").value_counts().reset_index()
            os_df.columns = ["os", "count"]
            st.plotly_chart(pie(os_df, "os", "count", "OS Split"), use_container_width=True)
    with col4:
        if "device_type" in seg_df.columns:
            dv = seg_df["device_type"].fillna("Unknown").value_counts().reset_index()
            dv.columns = ["device_type", "count"]
            st.plotly_chart(pie(dv, "device_type", "count", "Device Type"), use_container_width=True)

    if "app_version" in seg_df.columns:
        av = seg_df["app_version"].fillna("Unknown").value_counts().head(10).reset_index()
        av.columns = ["version", "count"]
        st.plotly_chart(bar(av, "version", "count", "App Version of Segment Users"),
                        use_container_width=True)

    st.divider()

    # ── Per-user activity table ───────────────────────────────────────────────
    st.markdown("#### Per-User Activity — inspect individual users")
    if "adid" in seg_df.columns:
        user_summary_parts = {"adid": seg_df["adid"].unique()}

        agg = {"adid": []}
        grp = seg_df.groupby("adid")

        rows = []
        for adid, g in grp:
            row = {"adid": adid}
            if "activity_kind" in g.columns:
                row["total_events"] = (g["activity_kind"] == "event").sum()
                row["total_sessions"] = (g["activity_kind"] == "session").sum()
            if "event_name" in g.columns:
                top_e = g[g.get("activity_kind", pd.Series()) == "event"]["event_name"].value_counts()
                row["top_event"] = top_e.index[0] if len(top_e) else ""
                row["unique_events"] = g["event_name"].nunique()
            if "country" in g.columns:
                row["country"] = g["country"].dropna().mode().iloc[0] if not g["country"].dropna().empty else ""
            if "os_name" in g.columns:
                row["os"] = g["os_name"].dropna().mode().iloc[0] if not g["os_name"].dropna().empty else ""
            if "app_version" in g.columns:
                row["app_version"] = g["app_version"].dropna().mode().iloc[0] if not g["app_version"].dropna().empty else ""
            if "lifetime_session_count" in g.columns:
                row["lifetime_sessions"] = pd.to_numeric(g["lifetime_session_count"], errors="coerce").max()
            if "time_spent" in g.columns:
                row["total_time_spent_s"] = pd.to_numeric(g["time_spent"], errors="coerce").sum()
            if "created_at" in g.columns and safe_date(g["created_at"]):
                row["first_seen"] = g["created_at"].min().date()
                row["last_seen"] = g["created_at"].max().date()
            rows.append(row)

        user_table = pd.DataFrame(rows).sort_values("total_events", ascending=False)
        st.dataframe(user_table, use_container_width=True, height=400)

        st.download_button(
            f"Download user list for {label}",
            data=user_table.to_csv(index=False).encode(),
            file_name=f"users_{label.replace(' ', '_')[:40]}.csv",
            mime="text/csv",
        )


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="16Arena — Adjust Analytics",
    page_icon="game_die",
    layout="wide",
)

# ── Authentication ─────────────────────────────────────────────────────────────
_creds = {
    "usernames": {
        username: {
            "name":     data["name"],
            "email":    data["email"],
            "password": data["password"],
        }
        for username, data in st.secrets["credentials"].items()
    }
}

_authenticator = stauth.Authenticate(
    _creds,
    st.secrets["cookie"]["name"],
    st.secrets["cookie"]["key"],
    st.secrets["cookie"]["expiry_days"],
)

_login_result = _authenticator.login()
_auth_status = (
    _login_result[1]
    if _login_result is not None
    else st.session_state.get("authentication_status")
)

if _auth_status is False:
    st.error("Incorrect username or password.")
    st.stop()
elif _auth_status is None:
    st.info("Please log in to access the dashboard.")
    st.stop()

# ── Authenticated ──────────────────────────────────────────────────────────────
st.title("16Arena — Adjust Analytics Dashboard")
st.caption("Live data from Azure Blob Storage · hourly Adjust raw exports")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    _authenticator.logout("Logout", "sidebar")
    st.divider()
    st.header("Filters")
    all_blobs = list_blob_names()

    def blob_date(name):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        return m.group(1) if m else "unknown"

    dates = sorted({blob_date(b) for b in all_blobs if blob_date(b) != "unknown"})
    selected_dates = st.multiselect(
        "Select dates", dates,
        default=dates[-min(7, len(dates)):],
    )

    filtered_blobs = (
        [b for b in all_blobs if blob_date(b) in selected_dates]
        if selected_dates else all_blobs
    )
    st.caption(f"{len(filtered_blobs)} blobs match")
    max_blobs = st.slider("Max blobs to load", 10, len(all_blobs), min(50, len(all_blobs)))
    filtered_blobs = filtered_blobs[:max_blobs]
    st.divider()
    st.caption(f"Total blobs in container: {len(all_blobs)}")
    if st.button("Clear data cache", help="Delete all cached parquet files and re-download on next load"):
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        load_data.clear()
        list_blob_names.clear()
        st.success("Cache cleared — reload the page to re-fetch.")

if not filtered_blobs:
    st.warning("No blobs match the selected filters.")
    st.stop()

raw = load_data(tuple(filtered_blobs))
if raw.empty:
    st.error("No data loaded. Check blob access.")
    st.stop()

df = clean(raw.copy())

# ── Global derived frames ──────────────────────────────────────────────────────
installs   = df[df["activity_kind"] == "install"]   if "activity_kind" in df.columns else pd.DataFrame()
events_df  = df[df["activity_kind"] == "event"]     if "activity_kind" in df.columns else pd.DataFrame()
sessions_df= df[df["activity_kind"] == "session"]   if "activity_kind" in df.columns else pd.DataFrame()
unique_users = df["adid"].nunique() if "adid" in df.columns else 0

# ── KPI strip ──────────────────────────────────────────────────────────────────
st.subheader("Overview")
c1, c2, c3, c4, c5, c6 = st.columns(6)
organic_pct = (
    installs["is_organic"].fillna(False).mean() * 100
    if "is_organic" in installs.columns and len(installs) else 0
)
with c1: kpi("Total Records",    f"{len(df):,}")
with c2: kpi("Unique Users",     f"{unique_users:,}")
with c3: kpi("Installs",         f"{len(installs):,}")
with c4: kpi("Events",           f"{len(events_df):,}")
with c5: kpi("Sessions",         f"{len(sessions_df):,}")
with c6: kpi("Organic Install %",f"{organic_pct:.1f}%")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Acquisition",
    "Retention & Sessions",
    "Events",
    "Attribution",
    "Geo & Device",
    "App Version",
    "Campaign / Network Deep Dive",
    "Raw Data",
])

# ─── TAB 1: Acquisition ────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Acquisition")
    if installs.empty:
        st.info("No install records in the selected period.")
    else:
        if "created_at" in installs.columns and safe_date(installs["created_at"]):
            inst_daily = (
                installs.assign(date=installs["created_at"].dt.date)
                .groupby("date").size().reset_index(name="installs")
            )
            st.plotly_chart(line(inst_daily, "date", "installs", "Daily Installs"),
                            use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "network_name" in installs.columns:
                net = installs["network_name"].fillna("Unknown").value_counts().head(10).reset_index()
                net.columns = ["network", "count"]
                st.plotly_chart(bar(net, "count", "network", "Installs by Network"),
                                use_container_width=True)
        with col2:
            if "country" in installs.columns:
                ctry = installs["country"].fillna("Unknown").value_counts().head(15).reset_index()
                ctry.columns = ["country", "count"]
                st.plotly_chart(bar(ctry, "count", "country", "Installs by Country"),
                                use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "campaign_name" in installs.columns:
                camp = (
                    installs["campaign_name"].fillna("Organic/Unknown")
                    .value_counts().head(10).reset_index()
                )
                camp.columns = ["campaign", "count"]
                st.plotly_chart(bar(camp, "count", "campaign", "Installs by Campaign"),
                                use_container_width=True)
        with col4:
            if "is_organic" in installs.columns:
                org = (
                    installs["is_organic"]
                    .map({True: "Organic", False: "Paid", None: "Unknown"})
                    .value_counts().reset_index()
                )
                org.columns = ["type", "count"]
                st.plotly_chart(pie(org, "type", "count", "Organic vs Paid"),
                                use_container_width=True)

# ─── TAB 2: Retention & Sessions ───────────────────────────────────────────────
with tabs[1]:
    st.header("Retention & Sessions")
    col1, col2 = st.columns(2)
    with col1:
        if "session_count" in df.columns and not df["session_count"].isna().all():
            sc = (df["session_count"].clip(upper=20).value_counts()
                  .sort_index().reset_index())
            sc.columns = ["sessions", "users"]
            st.plotly_chart(bar(sc, "sessions", "users",
                                "Session Count Distribution (capped 20)"),
                            use_container_width=True)
    with col2:
        if "lifetime_session_count" in df.columns:
            lsc = pd.to_numeric(df["lifetime_session_count"], errors="coerce").dropna()
            if len(lsc):
                buckets = pd.cut(lsc, bins=[0, 1, 3, 7, 14, 30, 9999],
                                 labels=["1", "2-3", "4-7", "8-14", "15-30", "30+"])
                lsc_df = buckets.value_counts().sort_index().reset_index()
                lsc_df.columns = ["lifetime_sessions", "count"]
                st.plotly_chart(bar(lsc_df, "lifetime_sessions", "count",
                                    "Lifetime Session Buckets"),
                                use_container_width=True)

    if "time_spent" in df.columns:
        ts = pd.to_numeric(df["time_spent"], errors="coerce").dropna()
        if len(ts):
            st.markdown(
                f"**Avg time spent:** {ts.mean():.1f}s  |  "
                f"**Median:** {ts.median():.1f}s  |  **Max:** {ts.max():.0f}s"
            )

    if "is_reattributed" in df.columns:
        reatt = df["is_reattributed"].value_counts().reset_index()
        reatt.columns = ["status", "count"]
        reatt["status"] = reatt["status"].map(
            {True: "Re-attributed", False: "New", None: "Unknown"}
        )
        col3, _ = st.columns(2)
        with col3:
            st.plotly_chart(pie(reatt, "status", "count", "Re-attribution vs New"),
                            use_container_width=True)

    if not sessions_df.empty and "created_at" in sessions_df.columns and safe_date(sessions_df["created_at"]):
        sess_daily = (
            sessions_df.assign(date=sessions_df["created_at"].dt.date)
            .groupby("date").size().reset_index(name="sessions")
        )
        st.plotly_chart(line(sess_daily, "date", "sessions", "Daily Sessions"),
                        use_container_width=True)

# ─── TAB 3: Events ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Events")
    if events_df.empty:
        st.info("No event records in the selected period.")
    else:
        if "event_name" in events_df.columns:
            top_events = events_df["event_name"].fillna("(blank)").value_counts().head(20).reset_index()
            top_events.columns = ["event", "count"]
            st.plotly_chart(bar(top_events, "count", "event", "Top 20 Events", h=500),
                            use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "event_name" in events_df.columns and "created_at" in events_df.columns and safe_date(events_df["created_at"]):
                top5 = events_df["event_name"].value_counts().head(5).index.tolist()
                ev_trend = (
                    events_df[events_df["event_name"].isin(top5)]
                    .assign(date=events_df["created_at"].dt.date)
                    .groupby(["date", "event_name"]).size().reset_index(name="count")
                )
                st.plotly_chart(line(ev_trend, "date", "count",
                                     "Top 5 Events Daily Trend", color="event_name"),
                                use_container_width=True)
        with col2:
            if "revenue" in events_df.columns:
                rev = pd.to_numeric(events_df["revenue"], errors="coerce").dropna()
                if len(rev) and rev.sum() > 0:
                    rev_ev = (
                        events_df.assign(revenue=pd.to_numeric(events_df["revenue"], errors="coerce"))
                        .groupby("event_name")["revenue"].sum()
                        .sort_values(ascending=False).head(10).reset_index()
                    )
                    st.plotly_chart(bar(rev_ev, "revenue", "event_name", "Revenue by Event"),
                                    use_container_width=True)

        if "adid" in events_df.columns:
            st.markdown("**Events per user stats:**")
            st.dataframe(
                events_df.groupby("adid")["event_name"].count()
                .describe().to_frame().T,
                use_container_width=True,
            )

# ─── TAB 4: Attribution ────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Attribution")
    col1, col2 = st.columns(2)
    with col1:
        if "network_name" in df.columns:
            net_all = df["network_name"].fillna("Unknown").value_counts().head(10).reset_index()
            net_all.columns = ["network", "count"]
            st.plotly_chart(bar(net_all, "count", "network", "All Activity by Network"),
                            use_container_width=True)
    with col2:
        if "tracker_name" in df.columns:
            tracker = df["tracker_name"].fillna("Unknown").value_counts().head(10).reset_index()
            tracker.columns = ["tracker", "count"]
            st.plotly_chart(bar(tracker, "count", "tracker", "Activity by Tracker"),
                            use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "first_tracker_name" in df.columns:
            ft = (
                df[["adid", "first_tracker_name"]].drop_duplicates("adid")
                ["first_tracker_name"].fillna("Unknown").value_counts().head(10).reset_index()
            )
            ft.columns = ["first_tracker", "users"]
            st.plotly_chart(bar(ft, "users", "first_tracker",
                                "Users by First Tracker (deduped)"),
                            use_container_width=True)
    with col4:
        if "is_organic" in df.columns:
            org_all = (
                df["is_organic"]
                .map({True: "Organic", False: "Paid", None: "Unknown"})
                .value_counts().reset_index()
            )
            org_all.columns = ["type", "count"]
            st.plotly_chart(pie(org_all, "type", "count", "Organic vs Paid (all activity)"),
                            use_container_width=True)

    if "adgroup_name" in df.columns:
        ag = df["adgroup_name"].fillna("N/A").value_counts().head(10).reset_index()
        ag.columns = ["adgroup", "count"]
        st.plotly_chart(bar(ag, "count", "adgroup", "Activity by Ad Group"),
                        use_container_width=True)

# ─── TAB 5: Geo & Device ───────────────────────────────────────────────────────
with tabs[4]:
    st.header("Geo & Device")
    col1, col2 = st.columns(2)
    with col1:
        if "country" in df.columns:
            ctry = df["country"].fillna("Unknown").value_counts().head(20).reset_index()
            ctry.columns = ["country", "count"]
            fig_map = px.choropleth(
                ctry, locations="country", locationmode="country names",
                color="count", title="Activity by Country",
                color_continuous_scale="Blues",
            )
            fig_map.update_layout(height=400, margin=dict(t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
    with col2:
        if "city" in df.columns:
            city = df["city"].fillna("Unknown").value_counts().head(15).reset_index()
            city.columns = ["city", "count"]
            st.plotly_chart(bar(city, "count", "city", "Top 15 Cities"),
                            use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "os_name" in df.columns:
            os_df = df["os_name"].fillna("Unknown").value_counts().reset_index()
            os_df.columns = ["os", "count"]
            st.plotly_chart(pie(os_df, "os", "count", "OS Distribution"),
                            use_container_width=True)
    with col4:
        if "device_type" in df.columns:
            dv = df["device_type"].fillna("Unknown").value_counts().reset_index()
            dv.columns = ["device_type", "count"]
            st.plotly_chart(pie(dv, "device_type", "count", "Device Type"),
                            use_container_width=True)

    if "connection_type" in df.columns:
        conn = df["connection_type"].fillna("Unknown").value_counts().reset_index()
        conn.columns = ["connection", "count"]
        st.plotly_chart(pie(conn, "connection", "count", "Connection Type"),
                        use_container_width=True)

# ─── TAB 6: App Version ────────────────────────────────────────────────────────
with tabs[5]:
    st.header("App Version")
    if "app_version" in df.columns:
        ver = df["app_version"].fillna("Unknown").value_counts().reset_index()
        ver.columns = ["version", "count"]
        st.plotly_chart(bar(ver, "version", "count", "Records by App Version"),
                        use_container_width=True)

        if "created_at" in df.columns and safe_date(df["created_at"]):
            ver_trend = (
                df.assign(date=df["created_at"].dt.date)
                .groupby(["date", "app_version"]).size().reset_index(name="count")
            )
            top_ver = df["app_version"].value_counts().head(5).index.tolist()
            ver_trend = ver_trend[ver_trend["app_version"].isin(top_ver)]
            st.plotly_chart(
                line(ver_trend, "date", "count",
                     "App Version Adoption Over Time", color="app_version"),
                use_container_width=True,
            )

    if "sdk_version" in df.columns:
        sdk = df["sdk_version"].fillna("Unknown").value_counts().reset_index()
        sdk.columns = ["sdk_version", "count"]
        st.plotly_chart(bar(sdk, "sdk_version", "count", "SDK Version Distribution"),
                        use_container_width=True)

# ─── TAB 7: Campaign / Network Deep Dive ───────────────────────────────────────
with tabs[6]:
    st.header("Campaign / Network Deep Dive")
    st.caption(
        "Select a segment below. The dashboard will find all users acquired "
        "from that source and show their full in-app behaviour."
    )

    # ── Segment picker ────────────────────────────────────────────────────────
    SEGMENT_COLS = {
        "Network":      "network_name",
        "Campaign":     "campaign_name",
        "Ad Group":     "adgroup_name",
        "Tracker":      "tracker_name",
        "Creative":     "creative_name",
        "First Tracker":"first_tracker_name",
    }

    available = {k: v for k, v in SEGMENT_COLS.items() if v in df.columns}
    if not available:
        st.warning("No attribution columns found in data.")
        st.stop()

    top_row = st.columns([2, 4, 1])
    with top_row[0]:
        seg_type = st.selectbox("Segment type", list(available.keys()))

    seg_col = available[seg_type]

    # Populate dropdown values from install rows first, then all rows
    source_df = installs if not installs.empty else df
    raw_vals = (
        source_df[seg_col]
        .dropna()
        .value_counts()
        .head(50)
        .index.tolist()
    )
    if not raw_vals:
        raw_vals = df[seg_col].dropna().unique().tolist()[:50]

    with top_row[1]:
        seg_value = st.selectbox(
            f"Select {seg_type}",
            raw_vals,
            help="Values ranked by install count",
        )

    # ── Compare toggle ────────────────────────────────────────────────────────
    with top_row[2]:
        compare_mode = st.checkbox("Compare 2nd segment")

    seg_value2 = None
    if compare_mode:
        remaining = [v for v in raw_vals if v != seg_value]
        seg_value2 = st.selectbox(f"2nd {seg_type} to compare", remaining)

    st.divider()

    # ── Build segment data ────────────────────────────────────────────────────
    # Strategy: get adids installed from that campaign, then pull all their activity
    def get_segment_users(col, value):
        """Return adids acquired via this segment."""
        if "adid" not in df.columns:
            return set()
        src = installs if not installs.empty else df
        if col not in src.columns:
            return set()
        return set(src.loc[src[col] == value, "adid"].dropna().unique())

    def get_segment_df(col, value):
        users = get_segment_users(col, value)
        if not users:
            # fallback: all rows matching the value (no install data)
            return df[df[col] == value].copy()
        return df[df["adid"].isin(users)].copy()

    seg_df1 = get_segment_df(seg_col, seg_value)

    if compare_mode and seg_value2:
        seg_df2 = get_segment_df(seg_col, seg_value2)

        # side-by-side comparison summary
        st.subheader("Side-by-Side Comparison")
        col_a, col_b = st.columns(2)

        def seg_summary(sdf, label):
            inst_n  = (sdf["activity_kind"] == "install").sum()  if "activity_kind" in sdf.columns else 0
            ev_n    = (sdf["activity_kind"] == "event").sum()    if "activity_kind" in sdf.columns else 0
            sess_n  = (sdf["activity_kind"] == "session").sum()  if "activity_kind" in sdf.columns else 0
            users_n = sdf["adid"].nunique()                       if "adid" in sdf.columns else 0
            avg_ev  = ev_n / users_n if users_n else 0
            avg_ses = sess_n / users_n if users_n else 0
            return pd.DataFrame({
                "Metric": ["Users", "Installs", "Events", "Sessions",
                           "Avg Events/User", "Avg Sessions/User"],
                label:    [users_n, inst_n, ev_n, sess_n,
                           round(avg_ev, 2), round(avg_ses, 2)],
            })

        summary1 = seg_summary(seg_df1, seg_value)
        summary2 = seg_summary(seg_df2, seg_value2)
        combined_summary = summary1.merge(summary2, on="Metric")
        st.dataframe(combined_summary, use_container_width=True, hide_index=True)

        # Event comparison bar
        if "event_name" in seg_df1.columns and "event_name" in seg_df2.columns:
            ev1 = (
                seg_df1[seg_df1.get("activity_kind", pd.Series()) == "event"]["event_name"]
                .value_counts().head(15).rename("count").reset_index()
                .assign(segment=seg_value)
            )
            ev2 = (
                seg_df2[seg_df2.get("activity_kind", pd.Series()) == "event"]["event_name"]
                .value_counts().head(15).rename("count").reset_index()
                .assign(segment=seg_value2)
            )
            ev_comp = pd.concat([ev1, ev2])
            ev_comp.columns = ["event", "count", "segment"]
            fig_comp = px.bar(
                ev_comp, x="count", y="event", color="segment",
                barmode="group", orientation="h",
                title="Event Comparison between segments",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_comp.update_layout(height=500, margin=dict(t=40, b=0))
            st.plotly_chart(fig_comp, use_container_width=True)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            render_segment_drill(seg_df1, df, seg_value)
        with col_b:
            render_segment_drill(seg_df2, df, seg_value2)
    else:
        render_segment_drill(seg_df1, df, seg_value)

# ─── TAB 8: Raw Data Explorer ──────────────────────────────────────────────────
with tabs[7]:
    st.header("Raw Data Explorer")

    show_cols = [c for c in df.columns if c not in PII_COLS]
    raw_view = df[show_cols].copy()

    # ── Column visibility ─────────────────────────────────────────────────────
    with st.expander("Choose columns to display", expanded=False):
        selected_cols = st.multiselect(
            "Columns", show_cols, default=show_cols,
            help="Uncheck columns you don't need"
        )
    if not selected_cols:
        selected_cols = show_cols
    raw_view = raw_view[selected_cols]

    # ── Global text search ────────────────────────────────────────────────────
    search_query = st.text_input(
        "Search anything",
        placeholder="Type any value to filter rows — searches all visible columns",
    )

    # ── Per-column filters ────────────────────────────────────────────────────
    with st.expander("Column filters", expanded=False):
        filter_cols = st.multiselect(
            "Pick columns to filter on",
            selected_cols,
            default=[],
        )

        col_filters = {}
        if filter_cols:
            fcols = st.columns(min(len(filter_cols), 3))
            for i, col in enumerate(filter_cols):
                with fcols[i % 3]:
                    series = raw_view[col]
                    numeric = pd.to_numeric(series, errors="coerce")
                    is_numeric = numeric.notna().mean() > 0.5
                    is_date = safe_date(series)

                    if is_date:
                        dates_in_col = series.dropna()
                        mn, mx = dates_in_col.min().date(), dates_in_col.max().date()
                        picked = st.date_input(f"{col} range", value=(mn, mx), key=f"flt_{col}")
                        if isinstance(picked, (list, tuple)) and len(picked) == 2:
                            col_filters[col] = ("date_range", picked[0], picked[1])
                    elif is_numeric:
                        lo = float(numeric.min())
                        hi = float(numeric.max())
                        if lo < hi:
                            picked_range = st.slider(
                                col, lo, hi, (lo, hi), key=f"flt_{col}"
                            )
                            col_filters[col] = ("numeric_range", picked_range[0], picked_range[1])
                        else:
                            st.caption(f"{col}: {lo} (single value)")
                    else:
                        unique_vals = sorted(series.dropna().unique().tolist())[:200]
                        picked_vals = st.multiselect(col, unique_vals, default=[], key=f"flt_{col}")
                        if picked_vals:
                            col_filters[col] = ("in", picked_vals)

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = raw_view.copy()

    # global text search across all visible columns
    if search_query.strip():
        q = search_query.strip().lower()
        mask = filtered.apply(
            lambda col: col.astype(str).str.lower().str.contains(q, na=False)
        ).any(axis=1)
        filtered = filtered[mask]

    # per-column filters
    for col, rule in col_filters.items():
        if col not in filtered.columns:
            continue
        kind = rule[0]
        if kind == "in":
            filtered = filtered[filtered[col].isin(rule[1])]
        elif kind == "numeric_range":
            numeric_col = pd.to_numeric(filtered[col], errors="coerce")
            filtered = filtered[(numeric_col >= rule[1]) & (numeric_col <= rule[2])]
        elif kind == "date_range":
            if safe_date(filtered[col]):
                dates_col = filtered[col].dt.date
                filtered = filtered[(dates_col >= rule[1]) & (dates_col <= rule[2])]

    # ── Summary + pagination ──────────────────────────────────────────────────
    total_rows = len(filtered)
    PAGE_SIZE = 500

    info_col, page_col, dl_col = st.columns([3, 2, 2])
    with info_col:
        st.markdown(
            f"**{total_rows:,} rows** match "
            f"({'all data' if not search_query and not col_filters else 'filtered'})"
            f" · {len(selected_cols)} columns"
        )

    total_pages = max(1, (total_rows - 1) // PAGE_SIZE + 1)
    with page_col:
        page = st.number_input(
            f"Page (of {total_pages})", min_value=1, max_value=total_pages, value=1, step=1
        )

    with dl_col:
        st.download_button(
            "Download filtered data as CSV",
            data=filtered.to_csv(index=False).encode(),
            file_name="adjust_filtered.csv",
            mime="text/csv",
        )

    # ── Table ─────────────────────────────────────────────────────────────────
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_df = filtered.iloc[start:end]

    st.dataframe(page_df, use_container_width=True, height=600)
    st.caption(
        f"Showing rows {start + 1}–{min(end, total_rows)} of {total_rows:,}  "
        f"· {PAGE_SIZE} rows per page"
    )

    # ── Column stats ──────────────────────────────────────────────────────────
    with st.expander("Column statistics for filtered data", expanded=False):
        stats_col = st.selectbox("Pick a column to inspect", selected_cols, key="stats_col")
        if stats_col:
            s = filtered[stats_col]
            numeric_s = pd.to_numeric(s, errors="coerce")
            is_num = numeric_s.notna().mean() > 0.5

            scol1, scol2 = st.columns(2)
            with scol1:
                st.markdown(f"**{stats_col}** — {len(s):,} rows, "
                            f"{s.isna().sum():,} nulls, "
                            f"{s.nunique():,} unique values")
                if is_num:
                    st.dataframe(numeric_s.describe().to_frame(), use_container_width=True)
                else:
                    top = s.fillna("(blank)").value_counts().head(20).reset_index()
                    top.columns = ["value", "count"]
                    st.dataframe(top, use_container_width=True)
            with scol2:
                if is_num:
                    fig_hist = px.histogram(
                        numeric_s.dropna(), nbins=30,
                        title=f"Distribution of {stats_col}",
                        color_discrete_sequence=["#4C78A8"],
                    )
                    fig_hist.update_layout(height=300, margin=dict(t=40, b=0))
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    top20 = s.fillna("(blank)").value_counts().head(20).reset_index()
                    top20.columns = ["value", "count"]
                    fig_bar = px.bar(
                        top20, x="count", y="value", orientation="h",
                        title=f"Top values — {stats_col}",
                        color_discrete_sequence=["#4C78A8"],
                    )
                    fig_bar.update_layout(height=350, margin=dict(t=40, b=0))
                    st.plotly_chart(fig_bar, use_container_width=True)
