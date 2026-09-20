"""Login/signup screen + logged-in sidebar (profile, chats, logout).

Session persistence: on successful login/signup we put a signed JWT into the
URL as `?s=<token>`. On every render we check `st.query_params` before showing
the login gate, so a page refresh keeps the user signed in. This works
instantly across refreshes (no async widget mount delay).

Tradeoff: the token is briefly visible in the URL — that's an intentional
security compromise for a personal project. The JWT is signed with a secret
derived from MONGO_URI, so forging one requires access to your Mongo creds.
"""

from __future__ import annotations

import streamlit as st

from finance_dashboard.auth import chats, profiles, session, users
from finance_dashboard.auth.db import get_db
from finance_dashboard.models import UserInputError

_USER_KEY = "auth_user"
_CHAT_ID_KEY = "current_chat_id"
_TOKEN_PARAM = "s"
_TOKEN_MAX_AGE_DAYS = 7


def _issue_session_token(username: str) -> None:
    """Store a signed JWT in the URL query params so refresh keeps the session."""
    token = session.encode_token(username, max_age_days=_TOKEN_MAX_AGE_DAYS)
    if not token:
        return
    try:
        st.query_params[_TOKEN_PARAM] = token
    except Exception:
        pass  # persistence failure must never break login itself


def _clear_session_token() -> None:
    try:
        if _TOKEN_PARAM in st.query_params:
            del st.query_params[_TOKEN_PARAM]
    except Exception:
        pass


def _restore_from_url() -> None:
    """If a valid session token is in ?s=, populate session_state[_USER_KEY]."""
    if _USER_KEY in st.session_state:
        return
    try:
        token = st.query_params.get(_TOKEN_PARAM)
    except Exception:
        return
    if not token:
        return
    username = session.decode_token(token)
    if username:
        st.session_state[_USER_KEY] = username


def current_user() -> str | None:
    return st.session_state.get(_USER_KEY)


def is_authenticated() -> bool:
    return current_user() is not None


def require_login() -> str | None:
    """Render login/signup gate. Returns username if logged in, else None (caller stops)."""
    _restore_from_url()
    if is_authenticated():
        return current_user()

    st.markdown("### Sign in to TradeMatrix")
    st.caption("Your chats, colors, and investment preferences are saved to your account.")

    db = get_db()
    if db is None:
        st.error(
            "**MongoDB not configured.** Add `MONGO_URI = \"mongodb+srv://…\"` "
            "to `.streamlit/secrets.toml`, then rerun the app. Auth is disabled "
            "until then to prevent unauthenticated access."
        )
        return None

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Log in"):
                try:
                    users.login(db, username, password)
                    uname = username.strip()
                    st.session_state[_USER_KEY] = uname
                    profiles.get_or_create(db, uname)  # default profile if none
                    _issue_session_token(uname)
                    st.rerun()
                except UserInputError as exc:
                    st.error(str(exc))

    with tab_signup:
        with st.form("signup_form"):
            username = st.text_input("Choose a username", key="signup_username")
            password = st.text_input("Choose a password (≥6 chars)", type="password",
                                     key="signup_password")
            if st.form_submit_button("Create account"):
                try:
                    users.signup(db, username, password)
                    uname = username.strip()
                    st.session_state[_USER_KEY] = uname
                    profiles.get_or_create(db, uname)
                    _issue_session_token(uname)
                    st.success("Account created — signed in.")
                    st.rerun()
                except UserInputError as exc:
                    st.error(str(exc))

    return None


def render_sidebar_account_panel() -> tuple[str, str]:
    """Sidebar: welcome banner, chats list, profile editor, logout.

    Returns (gain_color, loss_color) so the app can inject them as CSS variables.
    """
    username = current_user()
    db = get_db()
    if db is None or username is None:
        return "#00E676", "#FF3B30"

    profile = profiles.get_or_create(db, username)

    with st.sidebar:
        st.markdown(f"**Signed in as** `{username}`")
        if st.button("Log out", use_container_width=True):
            _clear_session_token()
            for k in (_USER_KEY, _CHAT_ID_KEY, "assistant_state", "assistant_history"):
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()

        # --- Chats ---
        st.subheader("Chats")
        if st.button("＋ New chat", use_container_width=True):
            _start_new_chat()
            st.rerun()

        chat_list = chats.list_for_user(db, username)
        if chat_list:
            with st.expander(f"📂 Load previous ({len(chat_list)})", expanded=False):
                for c in chat_list:
                    label = f"{c.get('title', '(untitled)')} · {c['updated_at']:%b %d %H:%M}"
                    if st.button(label, key=f"load-{c['_id']}", use_container_width=True):
                        _load_chat(db, username, c["_id"])
                        st.rerun()

        st.divider()

        # --- Profile: appearance ---
        # These control the two series colors on comparison charts (e.g., AAPL
        # vs MSFT). Gains and losses stay green/red automatically everywhere
        # else (percentages, direction-colored lines, correlation heatmap).
        st.subheader("Chart colors")
        st.caption("For comparison charts (2-series plots). Gains stay green, losses red.")
        c1, c2 = st.columns(2)
        gain = c1.color_picker("Color 1", value=profile["gain_color"], key="gain_color")
        loss = c2.color_picker("Color 2", value=profile["loss_color"], key="loss_color")
        if gain != profile["gain_color"] or loss != profile["loss_color"]:
            profiles.update(db, username, {"gain_color": gain, "loss_color": loss})

        # --- Profile: investment preferences ---
        with st.expander("Investment preferences"):
            prefs = profile.get("preferences", {})
            new = {}
            new["target_annual_return_pct"] = st.number_input(
                "Target annual return %",
                value=float(prefs.get("target_annual_return_pct") or 10.0),
                step=1.0, key="pref_return",
            )
            new["max_volatility_pct"] = st.number_input(
                "Max annual volatility %",
                value=float(prefs.get("max_volatility_pct") or 25.0),
                step=1.0, key="pref_vol",
            )
            new["risk_preference"] = st.selectbox(
                "Risk preference",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(prefs.get("risk_preference") or "medium"),
                key="pref_risk",
            )
            new["max_recommendations"] = int(st.number_input(
                "How many recommendations",
                value=int(prefs.get("max_recommendations") or 3),
                min_value=1, max_value=20, step=1, key="pref_count",
            ))
            new["sectors"] = st.multiselect(
                "Preferred sectors",
                ["Tech", "Medical", "Finance"],
                default=prefs.get("sectors") or [],
                key="pref_sectors",
            )
            if st.button("Save preferences", key="pref_save"):
                profiles.update(db, username, {f"preferences.{k}": v for k, v in new.items()})
                st.success("Preferences saved.")

    return gain, loss


def current_chat_id():
    return st.session_state.get(_CHAT_ID_KEY)


def ensure_chat(db, username: str, first_message: str | None = None):
    """Ensure a chat exists for this session. If not, create one titled from the first message."""
    if current_chat_id() is None:
        title = "New chat"
        if first_message:
            title = (first_message.strip().split("\n", 1)[0])[:60] or "New chat"
        st.session_state[_CHAT_ID_KEY] = chats.create(db, username, title=title)
    return st.session_state[_CHAT_ID_KEY]


def _start_new_chat() -> None:
    """Clear the in-memory chat state; a fresh chat doc is created on first message."""
    for k in ("assistant_state", "assistant_history", _CHAT_ID_KEY):
        st.session_state.pop(k, None)


def _load_chat(db, username: str, chat_id) -> None:
    """Load an existing chat's messages into session state for continued conversation."""
    from finance_dashboard.assistant.assistant import InvestmentAssistant

    doc = chats.load(db, username, chat_id)
    assistant = InvestmentAssistant()
    state = assistant.new_state()
    for msg in doc.get("messages", []):
        state.messages.append({"role": msg["role"], "content": msg["content"]})

    st.session_state["assistant_state"] = (assistant, state)
    st.session_state["assistant_history"] = [
        (m["role"], m["content"], None) for m in doc.get("messages", [])
    ]
    st.session_state[_CHAT_ID_KEY] = chat_id
