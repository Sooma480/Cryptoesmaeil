#!/usr/bin/env python3
"""
Crypto Signals Telegram Bot

- Gives LONG/SHORT suggestions for crypto futures/spot based on EMA cross + RSI filter + ATR-based risk.
- Works with any CCXT-supported exchange (default: Binance). For futures, set market="future" in /settings.
- Commands:
    /start – register & show help
    /help – show help
    /add <symbol> <timeframe> [exchange] – start watching a market (e.g., /add BTC/USDT 15m binance)
    /remove <symbol> <timeframe> – stop watching a market
    /list – list active watches
    /settings – show current settings
    /setrisk <percent> – risk per trade in % of equity (default 1)
    /setrr <rr1> <rr2> – risk:reward targets (default 1.5 3)
    /setequity <usd> – set notional equity to size positions (default 1000)
    /setmarket <spot|future> – toggle context for messages (does not place orders)
    /ping – health check

Notes:
- This bot ONLY sends signals. It does NOT place trades.
- Educational use only. Crypto is highly volatile. Do your own research.

Prereqs (Python 3.10+):
    pip install python-telegram-bot==21.6 ccxt pandas numpy ta aiohttp python-dotenv

Env vars (create a .env file alongside this script):
    TELEGRAM_BOT_TOKEN=123456:ABC...
    # Optional: restrict to your user id(s) (comma-separated). Find via @userinfobot
    ALLOWED_USER_IDS=123456789,987654321

Run:
    python bot.py

"""
from __future__ import annotations
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
from dotenv import load_dotenv

import ccxt.async_support as ccxt  # async version
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
)

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_IDS = {int(uid) for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip().isdigit()}

if not BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN missing. Put it in .env")

# ----------------------------- Data Models -----------------------------
@dataclass
class Watch:
    symbol: str
    timeframe: str = "15m"
    exchange_name: str = "binance"
    last_signal_ts: float = 0.0  # epoch seconds of last sent signal candle

@dataclass
class Settings:
    risk_percent: float = 1.0
    rr_targets: Tuple[float, float] = (1.5, 3.0)
    equity_usd: float = 1000.0
    market: str = "future"  # for message context only

@dataclass
class UserState:
    watches: Dict[Tuple[str, str], Watch] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)

# in-memory user states (user_id -> UserState)
USERS: Dict[int, UserState] = {}


# ----------------------------- Utils -----------------------------
TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440
}

def human(v: float, digits: int = 4) -> str:
    if v is None or np.isnan(v):
        return "-"
    # choose digits based on magnitude
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:,.3f}"
    return f"{v:,.6f}"


def position_size(equity: float, risk_pct: float, entry: float, stop: float) -> float:
    risk_amount = equity * (risk_pct / 100.0)
    distance = abs(entry - stop)
    if distance <= 0:
        return 0.0
    size = risk_amount / distance
    return max(size, 0.0)


# ----------------------------- Strategy -----------------------------
async def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 400) -> pd.DataFrame:
    data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(data, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def signal_from_df(df: pd.DataFrame):
    """Return (signal, entry, sl, tp1, tp2, meta) or (None, ...)
    signal in {"LONG", "SHORT"}
    """
    if df is None or len(df) < 200:
        return None, None, None, None, None, {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    ema50 = EMAIndicator(pd.Series(close), window=50).ema_indicator().values
    ema200 = EMAIndicator(pd.Series(close), window=200).ema_indicator().values
    rsi = RSIIndicator(pd.Series(close), window=14).rsi().values

    atr = AverageTrueRange(pd.Series(high), pd.Series(low), pd.Series(close), window=14).average_true_range().values

    # use last closed candle
    idx = -2  # second to last row is the last CLOSED candle
    c = close[idx]
    e50 = ema50[idx]
    e200 = ema200[idx]
    r = rsi[idx]
    a = atr[idx]

    meta = {"ema50": e50, "ema200": e200, "rsi": r, "atr": a, "close": c}

    long_cond = e50 > e200 and r > 55 and c > e50
    short_cond = e50 < e200 and r < 45 and c < e50

    if long_cond:
        entry = c
        sl = max(c - 2.0 * a, 0)
        rr1, rr2 = 1.5, 3.0
        tp1 = c + rr1 * (c - sl)
        tp2 = c + rr2 * (c - sl)
        return "LONG", entry, sl, tp1, tp2, meta
    if short_cond:
        entry = c
        sl = c + 2.0 * a
        rr1, rr2 = 1.5, 3.0
        tp1 = c - rr1 * (sl - c)
        tp2 = c - rr2 * (sl - c)
        return "SHORT", entry, sl, tp1, tp2, meta

    return None, None, None, None, None, meta


# ----------------------------- Exchange Factory -----------------------------
async def get_exchange(name: str):
    name = name.lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Exchange '{name}' not found in ccxt")
    cls = getattr(ccxt, name)
    ex = cls({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},  # futures-compatible symbols if available
    })
    await ex.load_markets()
    return ex


# ----------------------------- Bot Handlers -----------------------------
HELP_TEXT = (
    "أهلًا! أنا بوت توصيات عملات رقمية (تعليمي).\n\n"
    "الأوامر:\n"
    "/add <رمز> <فريم> [منصة] – ابدأ المراقبة (مثال: /add SOL/USDT 15m binance)\n"
    "/remove <رمز> <فريم> – أوقف المراقبة\n"
    "/list – قائمة المراقبة\n"
    "/settings – الإعدادات الحالية\n"
    "/setrisk <٪> – نسبة المخاطرة لكل صفقة (افتراضي 1٪)\n"
    "/setrr <هدف1> <هدف2> – نسب المخاطرة/العائد (افتراضي 1.5 3)\n"
    "/setequity <قيمة بالدولار> – رأس المال الافتراضي للحجم\n"
    "/setmarket <spot|future> – سياق الرسائل\n"
    "/ping – فحص\n\n"
    "⚠️ تنبيه: هذه ليست نصيحة استثمارية. قم دائمًا بإدارة المخاطر."
)


def ensure_user(update: Update) -> int:
    uid = update.effective_user.id
    if ALLOWED_IDS and uid not in ALLOWED_IDS:
        raise PermissionError("غير مصرح: أضف معرفك في ALLOWED_USER_IDS أو اتركه فارغًا للسماح للجميع.")
    if uid not in USERS:
        USERS[uid] = UserState()
    return uid


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ensure_user(update)
        await update.message.reply_text(HELP_TEXT)
    except PermissionError as e:
        await update.message.reply_text(str(e))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    args = context.args
    if len(args) < 2:
        return await update.message.reply_text("الاستخدام: /add <رمز> <فريم> [منصة]\nمثال: /add SOL/USDT 15m binance")
    symbol = args[0].upper()
    timeframe = args[1]
    exchange_name = args[2].lower() if len(args) >= 3 else "binance"

    USERS[uid].watches[(symbol, timeframe)] = Watch(symbol, timeframe, exchange_name)
    await update.message.reply_text(f"تمت إضافة {symbol} {timeframe} على {exchange_name}.")


async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    args = context.args
    if len(args) < 2:
        return await update.message.reply_text("الاستخدام: /remove <رمز> <فريم>")
    key = (args[0].upper(), args[1])
    if key in USERS[uid].watches:
        USERS[uid].watches.pop(key)
        await update.message.reply_text(f"تمت إزالة {key[0]} {key[1]}.")
    else:
        await update.message.reply_text("غير موجودة في المراقبة.")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    if not USERS[uid].watches:
        return await update.message.reply_text("لا توجد مراقبات. استخدم /add لإضافة.")
    lines = ["القائمة:"]
    for w in USERS[uid].watches.values():
        lines.append(f"• {w.symbol} {w.timeframe} @ {w.exchange_name}")
    await update.message.reply_text("\n".join(lines))


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    s = USERS[uid].settings
    await update.message.reply_text(
        f"الإعدادات الحالية:\n"
        f"• السوق: {s.market}\n"
        f"• المخاطرة/صفقة: {s.risk_percent}%\n"
        f"• RR: {s.rr_targets[0]} / {s.rr_targets[1]}\n"
        f"• رأس المال: ${human(s.equity_usd)}"
    )


async def cmd_setrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    if not context.args:
        return await update.message.reply_text("الاستخدام: /setrisk <نسبة بالمئة>")
    try:
        val = float(context.args[0])
    except Exception:
        return await update.message.reply_text("قيمة غير صالحة")
    USERS[uid].settings.risk_percent = max(0.01, min(val, 10.0))
    await update.message.reply_text(f"تم ضبط المخاطرة على {USERS[uid].settings.risk_percent}%.")


async def cmd_setrr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    if len(context.args) < 2:
        return await update.message.reply_text("الاستخدام: /setrr <هدف1> <هدف2>")
    try:
        r1 = float(context.args[0]); r2 = float(context.args[1])
    except Exception:
        return await update.message.reply_text("قيم غير صالحة")
    USERS[uid].settings.rr_targets = (max(0.5, r1), max(0.5, r2))
    await update.message.reply_text(f"تم ضبط RR على {r1} / {r2}.")


async def cmd_setequity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    if not context.args:
        return await update.message.reply_text("الاستخدام: /setequity <قيمة بالدولار>")
    try:
        val = float(context.args[0])
    except Exception:
        return await update.message.reply_text("قيمة غير صالحة")
    USERS[uid].settings.equity_usd = max(10.0, val)
    await update.message.reply_text(f"تم ضبط رأس المال على ${human(USERS[uid].settings.equity_usd)}.")


async def cmd_setmarket(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = ensure_user(update)
    if not context.args:
        return await update.message.reply_text("الاستخدام: /setmarket <spot|future>")
    val = context.args[0].lower()
    if val not in {"spot", "future"}:
        return await update.message.reply_text("يجب أن تكون spot أو future")
    USERS[uid].settings.market = val
    await update.message.reply_text(f"تم ضبط السوق إلى {val}.")


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_user(update)
    await update.message.reply_text("pong ✅")


# ----------------------------- Signal Loop -----------------------------
async def signal_loop(app: Application):
    # one shared exchange cache per exchange name
    exchanges: Dict[str, ccxt.Exchange] = {}

    while True:
        try:
            # iterate users and their watches
            for uid, state in USERS.items():
                for key, w in list(state.watches.items()):
                    try:
                        ex = exchanges.get(w.exchange_name)
                        if ex is None:
                            ex = await get_exchange(w.exchange_name)
                            exchanges[w.exchange_name] = ex
                        df = await fetch_ohlcv(ex, w.symbol, w.timeframe, limit=400)
                        if df.empty:
                            continue
                        # last closed candle time
                        last_closed_ts = df.iloc[-2]["timestamp"].value / 1e9
                        if last_closed_ts <= w.last_signal_ts:
                            continue  # already processed

                        sig, entry, sl, tp1, tp2, meta = signal_from_df(df)
                        w.last_signal_ts = last_closed_ts

                        if sig is None:
                            continue

                        # compute sizing
                        s = state.settings
                        size = position_size(s.equity_usd, s.risk_percent, entry, sl)

                        text = (
                            f"<b>📣 إشارة {sig} ({state.settings.market.upper()})</b>\n"
                            f"الرمز: <b>{w.symbol}</b> | الفريم: <b>{w.timeframe}</b> | المنصة: <b>{w.exchange_name}</b>\n"
                            f"السعر: <code>{human(meta.get('close'))}</code>\n"
                            f"EMA50: <code>{human(meta.get('ema50'))}</code> | EMA200: <code>{human(meta.get('ema200'))}</code> | RSI14: <code>{human(meta.get('rsi'))}</code>\n"
                            f"الدخول: <b>{human(entry)}</b>\n"
                            f"وقف الخسارة: <b>{human(sl)}</b>\n"
                            f"TP1 (RR {state.settings.rr_targets[0]}): <b>{human(tp1)}</b>\n"
                            f"TP2 (RR {state.settings.rr_targets[1]}): <b>{human(tp2)}</b>\n"
                            f"📏 حجم المركز التقريبي: <b>{human(size)}</b> وحدة\n"
                            f"⚠️ تعليمي فقط – ليست نصيحة استثمارية."
                        )

                        await app.bot.send_message(chat_id=uid, text=text, parse_mode=ParseMode.HTML)

                    except Exception as e:
                        await app.bot.send_message(chat_id=uid, text=f"خطأ أثناء تحليل {w.symbol} {w.timeframe}: {e}")
                        continue
        except Exception as outer:
            # log to first user if exists
            if USERS:
                any_uid = next(iter(USERS.keys()))
                await app.bot.send_message(chat_id=any_uid, text=f"حلقة الإشارات توقفت مؤقتًا: {outer}")
        # sleep minimal timeframe among active watches (default 60s)
        min_minutes = 60
        active_tfs = [TF_MINUTES.get(w.timeframe, 15) for st in USERS.values() for w in st.watches.values()]
        if active_tfs:
            min_minutes = max(1, min(active_tfs) // 2)  # half timeframe to catch new closes
        await asyncio.sleep(min_minutes * 60)


# ----------------------------- Main -----------------------------
async def main():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("add", cmd_add))
    application.add_handler(CommandHandler("remove", cmd_remove))
    application.add_handler(CommandHandler("list", cmd_list))
    application.add_handler(CommandHandler("settings", cmd_settings))
    application.add_handler(CommandHandler("setrisk", cmd_setrisk))
    application.add_handler(CommandHandler("setrr", cmd_setrr))
    application.add_handler(CommandHandler("setequity", cmd_setequity))
    application.add_handler(CommandHandler("setmarket", cmd_setmarket))
    application.add_handler(CommandHandler("ping", cmd_ping))

    # background signal loop
    asyncio.create_task(signal_loop(application))

    print("Bot running… Press Ctrl+C to stop.")
    await application.run_polling(close_loop=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped.")
