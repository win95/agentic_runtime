"""Manual end-to-end test for agent_lark_me (Lark DM + group + platform).

Chay tren may cua tran sau khi pull nhanh ``agent-lark-me``. Script nay
chi KIEM TRA — no khong sua code, khong seed, khong restart dich vu nao.
No goi 3 thu: Lark API (bang tenant token cua agent), platform API
(bang ``API_KEY``) va cac health endpoint tren localhost.

Chay::

    .venv/bin/python scripts/manual/lark_me_test.py --self-test
    .venv/bin/python scripts/manual/lark_me_test.py --check
    .venv/bin/python scripts/manual/lark_me_test.py --ping
    .venv/bin/python scripts/manual/lark_me_test.py --group
    .venv/bin/python scripts/manual/lark_me_test.py --ping --group --verbose

Khong co flag = ``--self-test``. Exit code: 0 khi moi buoc PASS (SKIP
khong tinh la loi), 1 khi co buoc FAIL, 2 khi thieu cau hinh / dung sai.

Dieu kien tien quyet (tren may tran, truoc khi chay):
- Da pull nhanh ``agent-lark-me``; ``make migrate-users`` (user_service
  len migration 0002) va ``make seed-agent AGENT=agent_lark_me
  MISSION=agent_lark_me_mission`` da chay; worker + platform API da
  restart de nhan mission moi.
- Platform API chay o :8000 (container ``api`` cua compose chinh hoac
  uvicorn tren host), user-service o :8001, frame o :8002 (health
  server; WS mode) — xem ``docker-compose.dev.yml``.
- Repo-root ``.env`` co ``LARK_APP_ID`` / ``LARK_APP_SECRET`` (cua
  agent), ``API_KEY`` (platform), ``LARK_ALLOWLIST`` (open_id cua tran).
  Script doc ``.env`` tu repo-root (resolve tu vi tri file, khong phu
  thuoc CWD — giong frame); shell env thang gia tri trong file.
- Bot da duoc them vao DM (tran chat DM voi bot) va vao group test
  (neu muon chay ``--group``); group chat_id dat trong
  ``LARK_TEST_GROUP_CHAT_ID``.

Cac buoc kiem tra:
1. ``--check``: frame ``GET /healthz`` + ``/readyz`` (:8002),
   platform ``GET /health`` (:8000), user-service ``GET /health``
   (:8001), va mission ``agent_lark_me_mission`` da duoc seed
   (``GET /missions/<id>``). In ro buoc nao fail + goi y ngan.
2. ``--ping``: doi tin ``ping`` trong DM voi bot, cho bot tra loi
   trong thread (chua ``pong``), roi tim execution tuong ung tren
   platform (theo ``session_id`` suy tu chat + user) va doi no
   SUCCEEDED.
3. ``--group``: kiem tra bot co trong group test khong, doi tin
   ``@bot ping`` trong group, cho reply trong thread + execution
   SUCCEEDED. Bo qua (SKIP) khi chua dat ``LARK_TEST_GROUP_CHAT_ID``.
4. ``--self-test``: chay 1 + 2 (+ 3 neu co group) lien mot mach.

LUU Y QUAN TRONG — vi sao --ping KHONG tu gui tin bang tenant token:
tin gui qua tenant token luon mang danh bot (sender_type ``app``) —
Lark khong gui event ``im.message.receive_v1`` cho chinh tin bot tu
gui, va frame cung drop tin do o rule 2 cua ``routing.route_message``
(``own_message``). Gui kieu do thi bot mai mai khong tra loi. Vi vay
``--ping`` chay o 1 trong 2 che do:
- Co ``LARK_USER_ACCESS_TOKEN`` (user token cua chinh tran, lay qua
  OAuth cua app nay): script tu gui ``ping`` vao DM nhu tran gui.
- Khong co: script in huong dan roi CHO tran gui ``ping`` tu Lark
  client trong vong ``--wait-ping`` giay (che do guided), sau do tu
  dong poll reply + execution. Day la cach duy nhat dung duoc voi
  chi tenant token; ``--group`` luon dung guided (tin gui qua API
  khong mang duoc @mention that nen frame se drop o buoc @mention).

GIA DINH (ghi lai vi mot agent khac dang sua code P0+P1 cung luc):
- ``session_id`` cua frame = ``"lark-" + sha1(chat_id + "|" +
  open_id)[:16]`` — script mirror cong thuc trong
  ``services/lark_frame/routing.py::session_id_for``. Neu cong thuc
  doi, sua ``session_id_for`` o duoi.
- Platform co ``GET /executions?session_id=...`` va
  ``GET /executions/{id}/result`` (da co trong ``app/routers/``).
- Agent tra loi dung thread bang ``lark_reply_message`` (theo skill
  ``agent-lark`` + system prompt). LUU Y: prompt/skill KHONG hardcode
  ``ping`` -> ``pong`` — check ``pong`` o day la theo quy uoc test;
  neu model tra loi kieu khac (vd "Minh day, ban can gi?") thi buoc
  reply se FAIL nhung script van in nguyen van reply de tran tu
  danh gia, va buoc execution SUCCEEDED van la tin hieu manh nhat
  rang pipeline chay duoc.

Khong bao phu: nut card (resolve/assign/escalate), luong alert tao
group, retrospective + doi ten group, rate limit / allowlist — nhung
thu do can thao tac tay tren Lark hoac script rieng.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _api import call as api_call  # noqa: E402
from _api import get_result, load_dotenv  # noqa: E402

LARK_DOMAIN = "https://open.larksuite.com"
MISSION_ID = "agent_lark_me_mission"
DEFAULT_PING_TEXT = "ping"
DEFAULT_WAIT_PING = 60.0  # cho tran gui "ping" (guided) / doi tin xuat hien
DEFAULT_WAIT_REPLY = 120.0  # cho bot reply (bao gom ca thoi gian agent chay)
DEFAULT_WAIT_EXEC = 180.0  # cho execution ve terminal
POLL_INTERVAL = 2.0


# --------------------------------------------------------------------------
# In an + tien ich chung
# --------------------------------------------------------------------------

@dataclass
class CheckRecorder:
    """Ghi PASS/FAIL/SKIP tung buoc + tom tat cuoi cung."""

    results: list[tuple[str, str, str]] = field(default_factory=list)

    def record(self, name: str, status: str, detail: str = "") -> bool:
        assert status in ("PASS", "FAIL", "SKIP")
        self.results.append((name, status, detail))
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        line = f"  [{mark}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)
        return status == "PASS"

    def summary(self) -> int:
        failed = [n for n, s, _ in self.results if s == "FAIL"]
        skipped = [n for n, s, _ in self.results if s == "SKIP"]
        passed = len(self.results) - len(failed) - len(skipped)
        print(
            f"\nTom tat: {passed} PASS, {len(failed)} FAIL, "
            f"{len(skipped)} SKIP / {len(self.results)} buoc."
        )
        if failed:
            print("FAIL: " + ", ".join(failed))
            return 1
        print("Tat ca buoc da chay deu PASS.")
        return 0


def _mask(value: str) -> str:
    """Chi hien 4 ky tu cuoi de debug su hien dien — khong bao gio in secret."""
    value = (value or "").strip()
    if not value:
        return "(trong)"
    if len(value) <= 4:
        return "(da co, ****)"
    return f"(da co, ...{value[-4:]})"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    body: dict | None = None,
    timeout: float = 10.0,
) -> tuple[int | None, dict | None, str]:
    """Mot HTTP call JSON bang urllib. Tra (status, payload, loi).

    ``status`` None = khong noi duoc (Connection refused / timeout...).
    Payload khong phai JSON -> None (khong phai loi). Khong log header
    chua secret — caller tu chiu trach nhiem truyen header da che.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode("utf-8") or "{}"), ""
        except (ValueError, UnicodeDecodeError):
            return exc.code, None, raw.decode("utf-8", errors="replace")[:200]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, None, str(exc)[:200]
    if not raw:
        return 200, None, ""
    try:
        return 200, json.loads(raw.decode("utf-8")), ""
    except (ValueError, UnicodeDecodeError):
        return 200, None, raw.decode("utf-8", errors="replace")[:200]


def session_id_for(chat_id: str, open_id: str) -> str:
    """Session id per (chat, user) — mirror routing.session_id_for."""
    digest = hashlib.sha1(f"{chat_id}|{open_id}".encode("utf-8")).hexdigest()
    return f"lark-{digest[:16]}"


# --------------------------------------------------------------------------
# Lark API (tenant token — stdlib urllib, nhu services/lark_frame/platform.py)
# --------------------------------------------------------------------------

class LarkApi:
    """Lark read (+ send-as-user tuy chon) bang tenant token cua agent."""

    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self._token: str | None = None
        self._token_expires_at = 0.0

    def tenant_token(self) -> str | None:
        if self._token and self._token_expires_at > time.time() + 60:
            return self._token
        status, payload, err = _http_json(
            "POST",
            f"{LARK_DOMAIN}/open-apis/auth/v3/tenant_access_token/internal",
            body={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=10.0,
        )
        if status != 200 or not isinstance(payload, dict) or payload.get("code") != 0:
            code = payload.get("code") if isinstance(payload, dict) else None
            print(f"    lay tenant token that bai: HTTP {status} code={code} {err}")
            return None
        token = payload.get("tenant_access_token") or ""
        if not token:
            return None
        self._token = token
        self._token_expires_at = time.time() + int(payload.get("expire") or 7200)
        return token

    def _auth_headers(self) -> dict | None:
        token = self.tenant_token()
        if token is None:
            return None
        return {"Authorization": f"Bearer {token}"}

    def get(self, path: str, timeout: float = 10.0) -> tuple[int | None, dict | None, str]:
        headers = self._auth_headers()
        if headers is None:
            return None, None, "khong lay duoc tenant token"
        return _http_json("GET", f"{LARK_DOMAIN}{path}", headers=headers, timeout=timeout)

    def bot_open_id(self) -> str | None:
        status, payload, _ = self.get("/open-apis/bot/v3/info")
        if status != 200 or not isinstance(payload, dict) or payload.get("code") != 0:
            return None
        data = payload.get("data") or {}
        bot = data.get("bot") or data
        return bot.get("open_id") or bot.get("openid") or None

    def list_chats(self, page_size: int = 50) -> list[dict] | None:
        """Danh sach chat bot dang o (GET /open-apis/im/v1/chats)."""
        status, payload, _ = self.get(
            f"/open-apis/im/v1/chats?page_size={page_size}"
        )
        if status != 200 or not isinstance(payload, dict) or payload.get("code") != 0:
            return None
        data = payload.get("data") or {}
        items = data.get("items") or []
        return items if isinstance(items, list) else []

    def chat_member_open_ids(self, chat_id: str) -> list[str] | None:
        """Open_id cac thanh vien cua 1 chat (de tim DM p2p)."""
        status, payload, _ = self.get(
            f"/open-apis/im/v1/chats/{chat_id}/members"
            "?user_id_type=open_id&page_size=50"
        )
        if status != 200 or not isinstance(payload, dict) or payload.get("code") != 0:
            return None
        data = payload.get("data") or {}
        out: list[str] = []
        for item in data.get("items") or []:
            if not isinstance(item, dict):
                continue
            mid = item.get("member_id") or (item.get("id") or {}).get("open_id") or ""
            if mid:
                out.append(mid)
        return out

    def bot_is_in_chat(self, chat_id: str) -> bool | None:
        """Bot co o trong chat khong (GET .../members/is_in_chat)."""
        status, payload, _ = self.get(
            f"/open-apis/im/v1/chats/{chat_id}/members/is_in_chat"
        )
        if status != 200 or not isinstance(payload, dict) or payload.get("code") != 0:
            return None
        data = payload.get("data") or {}
        return bool(data.get("is_in_chat"))

    def find_dm_chat_id(self, user_open_id: str) -> str | None:
        """Tim DM p2p giua tran va bot: chat co dung 2 thanh vien
        (tran + bot). Gioi han 20 chat de khoi spam API."""
        chats = self.list_chats()
        if chats is None:
            return None
        checked = 0
        for chat in chats:
            chat_id = (chat or {}).get("chat_id") or ""
            if not chat_id or checked >= 20:
                continue
            checked += 1
            members = self.chat_member_open_ids(chat_id)
            if members is None:
                continue
            if len(members) == 2 and user_open_id in members:
                return chat_id
        return None


def _platform_get(path: str) -> tuple[int | None, dict | list | None]:
    """``_api.call`` an toan khi platform chet: ``_api.call`` raise
    URLError khi khong noi duoc (no khong tra status) — o day tra
    ``(None, None)`` de caller ghi FAIL thay vi crash."""
    try:
        return api_call("GET", path, log=False)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, None


def _message_text(item: dict) -> str:
    """Text cua 1 tin (content la chuoi JSON {"text": ...})."""
    if (item or {}).get("msg_type") != "text":
        return ""
    content = (item or {}).get("body") or {}
    raw = content.get("content") or ""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    if isinstance(parsed, dict):
        return str(parsed.get("text") or "")
    return ""


def _message_sender(item: dict) -> tuple[str, str]:
    """(open_id, sender_type) cua 1 tin trong history."""
    sender = (item or {}).get("sender") or {}
    sid = sender.get("id") or {}
    return sid.get("open_id") or "", sender.get("sender_type") or ""


def list_chat_messages(
    lark: LarkApi, chat_id: str, limit: int = 20
) -> list[dict] | None:
    """History moi nhat cua 1 chat (ByCreateTimeDesc, khong dao).

    Tra list item tho tu API: message_id, root_id, create_time (ms,
    dang chuoi), sender, body, msg_type. None = goi API that bai.
    """
    status, payload, _ = lark.get(
        "/open-apis/im/v1/messages?container_id_type=chat"
        f"&container_id={chat_id}&sort_type=ByCreateTimeDesc&page_size={limit}"
    )
    if status != 200 or not isinstance(payload, dict) or payload.get("code") != 0:
        return None
    data = payload.get("data") or {}
    items = data.get("items") or []
    return items if isinstance(items, list) else []


def send_as_user(
    user_token: str, *, receive_id: str, receive_id_type: str, text: str
) -> str | None:
    """Gui 1 tin text NHU NGUOI DUNG (user access token, khong phai
    tenant token). Tra message_id, None khi that bai. Chi dung cho DM
    (khong gui group — tin API khong mang @mention that)."""
    status, payload, err = _http_json(
        "POST",
        f"{LARK_DOMAIN}/open-apis/im/v1/messages?receive_id_type={receive_id_type}",
        headers={"Authorization": f"Bearer {user_token}"},
        body={
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
            "uuid": uuid.uuid4().hex,
        },
        timeout=10.0,
    )
    if status != 200 or not isinstance(payload, dict) or payload.get("code") != 0:
        code = payload.get("code") if isinstance(payload, dict) else None
        print(f"    gui bang user token that bai: HTTP {status} code={code} {err}")
        return None
    data = payload.get("data") or {}
    return data.get("message_id") or None


# --------------------------------------------------------------------------
# Buoc 1 — --check: health + mission da seed
# --------------------------------------------------------------------------

def check_health(rec: CheckRecorder, frame_url: str, api_url: str, user_url: str) -> bool:
    print("Buoc 1 — suc khoe dich vu:")
    ok = True
    status, payload, err = _http_json("GET", f"{frame_url}/healthz", timeout=5.0)
    ok &= rec.record(
        "frame /healthz (frame dang chay)",
        "PASS" if status == 200 else "FAIL",
        f"HTTP {status}" if status else f"khong noi duoc: {err} — chay frame chua? "
        "(`.venv/bin/python -m services.lark_frame` hoac "
        "`docker compose -f docker-compose.dev.yml up -d lark-frame`)",
    )
    status, payload, err = _http_json("GET", f"{frame_url}/readyz", timeout=5.0)
    ok &= rec.record(
        "frame /readyz (frame thay platform)",
        "PASS" if status == 200 else "FAIL",
        f"HTTP {status}"
        if status
        else f"khong noi duoc: {err} — frame chua chay?",
    )
    if status == 503:
        print("    goi y: /readyz 503 = frame len nhung platform :8000 khong tra loi.")
    status, payload, err = _http_json("GET", f"{api_url}/health", timeout=5.0)
    ok &= rec.record(
        "platform /health (:8000)",
        "PASS" if status == 200 else "FAIL",
        f"HTTP {status}" if status else f"khong noi duoc: {err} — chay platform chua?",
    )
    status, payload, err = _http_json("GET", f"{user_url}/health", timeout=5.0)
    ok &= rec.record(
        "user-service /health (:8001)",
        "PASS" if status == 200 else "FAIL",
        f"HTTP {status}"
        if status
        else f"khong noi duoc: {err} — chay user-service chua? "
        "(`docker compose -f docker-compose.dev.yml up -d user-service` "
        "hoac `make run-user-service`)",
    )
    status, payload = _platform_get(f"/missions/{MISSION_ID}")
    detail = ""
    if status == 200:
        detail = f"agent={(payload or {}).get('agent_id')}"
    elif status == 404:
        detail = (
            "mission chua co trong DB — chay `make seed-agent "
            "AGENT=agent_lark_me MISSION=agent_lark_me_mission` roi restart worker"
        )
    elif status is None:
        detail = "khong noi duoc platform"
    else:
        detail = f"HTTP {status}"
    ok &= rec.record(
        f"mission {MISSION_ID} da seed",
        "PASS" if status == 200 else "FAIL",
        detail,
    )
    return bool(ok)


# --------------------------------------------------------------------------
# Buoc 2/3 — --ping / --group: doi tin + reply + execution
# --------------------------------------------------------------------------

def _wait_for_ping_message(
    lark: LarkApi,
    chat_id: str,
    *,
    text: str,
    sender_open_id: str,
    since_ms: int,
    timeout: float,
    guided_prompt: str,
) -> dict | None:
    """Doi 1 tin `text` tu sender trong chat. Tra item do, None khi het gio.

    O che do guided, in huong dan 1 lan roi poll history — tran gui tin
    tu Lark client cua minh trong luc script doi.
    """
    print(f"    {guided_prompt}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        items = list_chat_messages(lark, chat_id, limit=20)
        if items:
            for item in items:
                create_ms = 0
                try:
                    create_ms = int(item.get("create_time") or 0)
                except (TypeError, ValueError):
                    create_ms = 0
                if create_ms < since_ms:
                    continue
                sender_id, _ = _message_sender(item)
                if sender_id != sender_open_id:
                    continue
                if _message_text(item).strip().lower() == text.strip().lower():
                    return item
        time.sleep(POLL_INTERVAL)
    return None


def _wait_for_bot_reply(
    lark: LarkApi,
    chat_id: str,
    *,
    ping_message_id: str,
    bot_open_id: str,
    timeout: float,
) -> tuple[dict | None, str]:
    """Doi bot reply trong thread cua tin ping, noi dung chua "pong".

    Tra (item, reply_text). Het gio -> (None, "").
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        items = list_chat_messages(lark, chat_id, limit=20)
        if items:
            for item in items:
                sender_id, _ = _message_sender(item)
                if sender_id != bot_open_id:
                    continue
                root_id = item.get("root_id") or item.get("parent_id") or ""
                if root_id != ping_message_id:
                    continue
                text = _message_text(item)
                if "pong" in text.lower():
                    return item, text
                # Co reply trong thread nhung khong co "pong" — ghi lai
                # de cuoi buoc in ra (model khong hardcode ping->pong).
                return item, text
        time.sleep(POLL_INTERVAL)
    return None, ""


def _wait_for_execution(
    session_id: str, *, known_ids: set[str], timeout: float
) -> dict | None:
    """Doi 1 execution MOI (chua co trong known_ids) cua session nay ve
    terminal. Tra execution dict, None khi het gio."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, payload = _platform_get(
            f"/executions?session_id={session_id}&limit=10"
        )
        if status == 200 and isinstance(payload, list):
            for exc in payload:
                if not isinstance(exc, dict):
                    continue
                if exc.get("mission_id") != MISSION_ID:
                    continue
                eid = exc.get("execution_id") or ""
                if not eid or eid in known_ids:
                    continue
                if exc.get("status") in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    return exc
        time.sleep(POLL_INTERVAL)
    return None


def _known_execution_ids(session_id: str) -> set[str]:
    """Id cac execution da co cua session — de phan biet run moi."""
    status, payload = _platform_get(
        f"/executions?session_id={session_id}&limit=10"
    )
    if status != 200 or not isinstance(payload, list):
        return set()
    return {
        (e or {}).get("execution_id") or ""
        for e in payload
        if isinstance(e, dict) and e.get("mission_id") == MISSION_ID
    } - {""}


def _run_ping_flow(
    rec: CheckRecorder,
    lark: LarkApi,
    *,
    label: str,
    chat_id: str,
    chat_desc: str,
    user_open_id: str,
    bot_open_id: str,
    text: str,
    wait_ping: float,
    wait_reply: float,
    wait_exec: float,
    user_token: str | None,
    send_via_user_token: bool,
) -> bool:
    """Mot luot ping day du: doi tin -> doi reply -> doi execution.

    ``send_via_user_token`` chi True cho DM (group luon guided — xem
    docstring dau file). Tra True khi tat ca PASS.
    """
    print(f"{label} — {chat_desc} (chat_id=...{chat_id[-6:]})")
    since_ms = int(time.time() * 1000) - 5000
    session_id = session_id_for(chat_id, user_open_id)
    known_ids = _known_execution_ids(session_id)
    ping_item: dict | None = None

    if send_via_user_token and user_token:
        print(f"    gui '{text}' vao DM bang user token (tu dong)...")
        message_id = send_as_user(
            user_token,
            receive_id=bot_open_id,
            receive_id_type="open_id",
            text=text,
        )
        if message_id:
            ping_item = {"message_id": message_id}
            print(f"    da gui, message_id=...{message_id[-8:]}")
        else:
            print("    gui tu dong that bai — chuyen sang cho tran gui tay.")
    if ping_item is None:
        if send_via_user_token:
            prompt = (
                f"Hay mo Lark va gui tin '{text}' vao DM voi bot ngay bay gio "
                f"(trong {int(wait_ping)}s)..."
            )
        else:
            prompt = (
                f"Hay mo Lark va gui tin '{text}' {chat_desc} ngay bay gio "
                f"(trong {int(wait_ping)}s)..."
            )
        ping_item = _wait_for_ping_message(
            lark,
            chat_id,
            text=text,
            sender_open_id=user_open_id,
            since_ms=since_ms,
            timeout=wait_ping,
            guided_prompt=prompt,
        )
    ok = rec.record(
        f"{label}: tin '{text}' toi bot",
        "PASS" if ping_item is not None else "FAIL",
        f"message_id=...{(ping_item or {}).get('message_id', '')[-8:]}"
        if ping_item
        else f"het {int(wait_ping)}s khong thay tin '{text}' cua ban trong chat "
        "— kiem tra: bot da duoc them vao chat chua? tin gui dung chat chua?",
    )
    if ping_item is None:
        return False
    ping_message_id = (ping_item or {}).get("message_id") or ""

    reply_item, reply_text = _wait_for_bot_reply(
        lark,
        chat_id,
        ping_message_id=ping_message_id,
        bot_open_id=bot_open_id,
        timeout=wait_reply,
    )
    has_pong = reply_item is not None and "pong" in reply_text.lower()
    detail = ""
    if reply_item is not None:
        shown = reply_text[:200].replace("\n", " ")
        detail = f"bot tra loi: {shown!r}"
        if not has_pong:
            detail += " (khong co 'pong' — prompt/skill khong hardcode ping->pong, xem docstring)"
    else:
        detail = (
            f"het {int(wait_reply)}s khong thay reply trong thread — kiem tra log "
            "frame (event co route khong?) va worker (execution co chay khong?)"
        )
    ok &= rec.record(
        f"{label}: bot tra loi trong thread (chua 'pong')",
        "PASS" if has_pong else "FAIL",
        detail,
    )

    exc = _wait_for_execution(
        session_id, known_ids=known_ids, timeout=wait_exec
    )
    if exc is None:
        rec.record(
            f"{label}: execution SUCCEEDED",
            "FAIL",
            f"het {int(wait_exec)}s khong thay execution moi cua session ve "
            "terminal — kiem tra worker co chay khong, mission da seed chua",
        )
        return False
    eid = exc.get("execution_id") or ""
    status = exc.get("status") or ""
    detail = f"execution_id={eid} status={status}"
    if exc.get("error"):
        detail += f" error={str(exc.get('error'))[:200]}"
    try:
        result = get_result(eid)
        output = (result or {}).get("output") or {}
        answer = output.get("answer") or ""
        if answer:
            detail += f" answer={str(answer)[:200]!r}"
        actions = output.get("actions_taken")
        if actions:
            detail += f" actions_taken={actions}"
    except Exception as exc2:  # noqa: BLE001 — result chi de hien thi
        detail += f" (khong doc duoc /result: {exc2})"
    ok &= rec.record(
        f"{label}: execution SUCCEEDED",
        "PASS" if status == "SUCCEEDED" else "FAIL",
        detail,
    )
    return bool(ok)


def check_ping(
    rec: CheckRecorder,
    lark: LarkApi,
    *,
    user_open_id: str,
    bot_open_id: str,
    text: str,
    wait_ping: float,
    wait_reply: float,
    wait_exec: float,
    user_token: str | None,
    dm_chat_id: str | None,
) -> bool:
    """--ping: DM voi bot."""
    print("Buoc 2 — ping DM:")
    if not dm_chat_id:
        print("    dang tim DM p2p giua ban va bot...")
        dm_chat_id = lark.find_dm_chat_id(user_open_id)
    if not dm_chat_id:
        rec.record(
            "ping DM: tim DM chat",
            "FAIL",
            "khong tim duoc DM p2p — dat LARK_TEST_DM_CHAT_ID trong .env "
            "(lay chat_id tu Lark client: mo DM voi bot, xem thong tin chat) "
            "hoac chat DM voi bot 1 tin truoc roi chay lai",
        )
        return False
    return _run_ping_flow(
        rec,
        lark,
        label="ping DM",
        chat_id=dm_chat_id,
        chat_desc="DM voi bot",
        user_open_id=user_open_id,
        bot_open_id=bot_open_id,
        text=text,
        wait_ping=wait_ping,
        wait_reply=wait_reply,
        wait_exec=wait_exec,
        user_token=user_token,
        send_via_user_token=True,
    )


def check_group(
    rec: CheckRecorder,
    lark: LarkApi,
    *,
    user_open_id: str,
    bot_open_id: str,
    text: str,
    wait_ping: float,
    wait_reply: float,
    wait_exec: float,
    group_chat_id: str | None,
) -> bool:
    """--group: @bot ping trong group test (luon guided)."""
    print("Buoc 3 — ping group:")
    if not group_chat_id:
        rec.record(
            "ping group",
            "SKIP",
            "chua dat LARK_TEST_GROUP_CHAT_ID trong .env — bo qua buoc group",
        )
        return True
    in_chat = lark.bot_is_in_chat(group_chat_id)
    if in_chat is False:
        rec.record(
            "ping group: bot co trong group",
            "FAIL",
            "bot khong o trong group — them bot vao group trong Lark client "
            "truoc khi chay (them thanh vien / @bot vao group)",
        )
        return False
    if in_chat is None:
        print("    (khong kiem tra duoc bot co trong group khong — van thu ping)")
    return _run_ping_flow(
        rec,
        lark,
        label="ping group",
        chat_id=group_chat_id,
        chat_desc="group test",
        user_open_id=user_open_id,
        bot_open_id=bot_open_id,
        text=text,
        wait_ping=wait_ping,
        wait_reply=wait_reply,
        wait_exec=wait_exec,
        user_token=None,
        send_via_user_token=False,
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _resolve_test_user() -> tuple[str | None, str]:
    """open_id nguoi test: LARK_TEST_USER_OPEN_ID, khong thi phan tu
    dau cua LARK_ALLOWLIST. Tra (open_id, cach lay)."""
    explicit = _env("LARK_TEST_USER_OPEN_ID")
    if explicit:
        return explicit, "LARK_TEST_USER_OPEN_ID"
    allowlist = [p.strip() for p in _env("LARK_ALLOWLIST").split(",") if p.strip()]
    if allowlist:
        return allowlist[0], "LARK_ALLOWLIST[0]"
    return None, ""


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Manual test end-to-end agent_lark_me (Lark DM/group + platform).",
    )
    parser.add_argument("--check", action="store_true", help="chi chay buoc 1 (health + seed)")
    parser.add_argument("--ping", action="store_true", help="chi chay buoc 2 (ping DM)")
    parser.add_argument("--group", action="store_true", help="chi chay buoc 3 (ping group)")
    parser.add_argument("--self-test", action="store_true", help="chay 1+2 (+3 neu co group)")
    parser.add_argument("--api-url", default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
                        help="platform API base URL (mac dinh %(default)s)")
    parser.add_argument("--frame-url", default="http://localhost:8002",
                        help="frame health base URL (mac dinh %(default)s)")
    parser.add_argument("--user-service-url", default="http://localhost:8001",
                        help="user-service base URL (mac dinh %(default)s)")
    parser.add_argument("--text", default=DEFAULT_PING_TEXT,
                        help="tin gui di de test (mac dinh %(default)s)")
    parser.add_argument("--wait-ping", type=float, default=DEFAULT_WAIT_PING,
                        help="cho tran gui tin toi da bao nhieu giay (mac dinh %(default)s)")
    parser.add_argument("--wait-reply", type=float, default=DEFAULT_WAIT_REPLY,
                        help="cho bot reply toi da bao nhieu giay (mac dinh %(default)s)")
    parser.add_argument("--wait-exec", type=float, default=DEFAULT_WAIT_EXEC,
                        help="cho execution ve terminal toi da (mac dinh %(default)s)")
    parser.add_argument("--verbose", action="store_true", help="bat log HTTP cua _api")
    args = parser.parse_args(argv)

    if not args.verbose:
        os.environ["MANUAL_LOG"] = "0"
    if args.api_url:
        os.environ["API_BASE_URL"] = args.api_url.rstrip("/")

    run_check = args.check
    run_ping = args.ping
    run_group = args.group
    if args.self_test or (not run_check and not run_ping and not run_group):
        run_check = run_ping = run_group = True

    rec = CheckRecorder()
    lark: LarkApi | None = None
    if run_ping or run_group:
        app_id = _env("LARK_APP_ID")
        app_secret = _env("LARK_APP_SECRET")
        api_key = _env("API_KEY")
        missing = [
            name
            for name, val in (
                ("LARK_APP_ID", app_id),
                ("LARK_APP_SECRET", app_secret),
                ("API_KEY", api_key),
            )
            if not val
        ]
        if missing:
            print(f"Thieu trong .env: {', '.join(missing)} — bo sung roi chay lai.")
            return 2
        print(
            f"Cau hinh: LARK_APP_ID={_mask(app_id)} API_KEY={_mask(api_key)} "
            f"LARK_APP_SECRET={_mask(app_secret)}"
        )
        lark = LarkApi(app_id, app_secret)
        user_open_id, how = _resolve_test_user()
        if not user_open_id:
            print(
                "Khong biet open_id nguoi test — dat LARK_TEST_USER_OPEN_ID hoac "
                "LARK_ALLOWLIST trong .env."
            )
            return 2
        print(f"Nguoi test: open_id=...{user_open_id[-6:]} (tu {how})")
        bot_open_id = lark.bot_open_id()
        if not bot_open_id:
            print(
                "Khong lay duoc bot open_id qua bot/v3/info — kiem tra "
                "LARK_APP_ID/LARK_APP_SECRET va Bot da bat trong console chua."
            )
            return 2
        print(f"Bot: open_id=...{bot_open_id[-6:]}")
        user_token = _env("LARK_USER_ACCESS_TOKEN") or None
        if user_token:
            print("LARK_USER_ACCESS_TOKEN: da co — --ping se tu gui tin.")
        else:
            print(
                "LARK_USER_ACCESS_TOKEN: chua co — --ping chay che do guided "
                "(ban gui tin tu Lark client)."
            )

    try:
        if run_check:
            check_health(
                rec,
                args.frame_url.rstrip("/"),
                args.api_url.rstrip("/"),
                args.user_service_url.rstrip("/"),
            )
        if run_ping and lark is not None:
            check_ping(
                rec,
                lark,
                user_open_id=user_open_id,  # type: ignore[arg-type]
                bot_open_id=bot_open_id,  # type: ignore[arg-type]
                text=args.text,
                wait_ping=args.wait_ping,
                wait_reply=args.wait_reply,
                wait_exec=args.wait_exec,
                user_token=user_token,  # type: ignore[arg-type]
                dm_chat_id=_env("LARK_TEST_DM_CHAT_ID") or None,
            )
        if run_group and lark is not None:
            check_group(
                rec,
                lark,
                user_open_id=user_open_id,  # type: ignore[arg-type]
                bot_open_id=bot_open_id,  # type: ignore[arg-type]
                text=args.text,
                wait_ping=args.wait_ping,
                wait_reply=args.wait_reply,
                wait_exec=args.wait_exec,
                group_chat_id=_env("LARK_TEST_GROUP_CHAT_ID") or None,
            )
    except KeyboardInterrupt:
        print("\nDung boi nguoi dung (Ctrl-C).")
        return 1
    return rec.summary()


if __name__ == "__main__":
    raise SystemExit(main())
