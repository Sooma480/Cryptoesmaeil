# main.py
"""
بوت تليجرام بسيط لعمل توصيات فيوتشر قصيرة (short) باستخدام بيانات Binance
(ويحاول استخدام CoinMarketCap لو حطيت CMC_API_KEY).
يعمل مسح دوري ويبعث توصيات إلى CHAT_ID مسجّل في المتغيرات البيئية.
"""

import os
import time
import threading
import requests
import math
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# ------- إعداد المتغيرات من Environment -------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
# يمكنك وضع معرف الشات هنا أو تسجيله بإرسال /start للبوت
CHAT_ID = os.environ.get("CHAT_ID")  # optional
CMC_API_KEY = os.environ.get("CMC_API_KEY")  # optional, لو عندك
INTERVAL_MIN = int(os.environ.get("INTERVAL_MIN", "15"))  # كل كم دقيقة يعمل scan
SLEEP_SECONDS = max(30, INTERVAL_MIN * 60)
SYMBOL_LIMIT = int(os.environ.get("SYMBOL_LIMIT", "20"))  # كم عملة يفحص كل جولة
KLINE_INTERVAL = os.environ.get("KLINE_INTERVAL", "1h")  # '5m','15m','1h','4h' الخ
WINDOW = int(os.environ.get("WINDOW", "100"))  # عدد الشموع لجلبها

if not TELEGRAM_TOKEN:
    raise Exception("الـ TELEGRAM_TOKEN غير موجود في متغيرات البيئة (Environment Variables).")

bot = Bot(token=TELEGRAM_TOKEN)

# ----------- Helpers for indicators -------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd_series(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def rsi_series(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_swing_high_low(series, lookback=20):
    # بسيط: أعلى وأدنى سعر في آخر lookback شموع
    recent = series[-lookback:]
    return recent.max(), recent.min()

def fib_levels(high, low):
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100.0%": low
    }
    return levels

# ------------ Binance data functions --------------
BINANCE_REST = "https://api.binance.com"

def fetch_binance_klines(symbol, interval=KLINE_INTERVAL, limit=WINDOW):
    url = f"{BINANCE_REST}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    # convert to DataFrame with close price and time
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_av","trades","tb_base_av","tb_quote_av","ignore"
    ])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

def get_top_symbols_from_binance(limit=SYMBOL_LIMIT):
    # جلب كل الـ tickers ثم نفلتر على USDT pairs ونرتب حسب حجم التداول
    url = f"{BINANCE_REST}/api/v3/ticker/24hr"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    # اختر USDT pairs فقط
    usdt = [d for d in data if d['symbol'].endswith('USDT') and 'DOWN' not in d['symbol'] and 'UP' not in d['symbol']]
    # ترتيب حسب quoteVolume
    usdt_sorted = sorted(usdt, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    symbols = [d['symbol'] for d in usdt_sorted[:limit]]
    return symbols

def get_symbols_from_cmc(limit=SYMBOL_LIMIT):
    # لو محطوط CMC_API_KEY: جلب أول العملات وترجمتها للـ symbol/USDT إن أمكن
    if not CMC_API_KEY:
        return []
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
        params = {"start": "1", "limit": str(limit), "convert": "USD"}
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get('data', [])
        # حاول تحويل كل اسم إلى SYMBOLUSDT عبر سؤال Binance عن وجود زوج
        symbols = []
        for item in data:
            sym = item.get('symbol')
            if not sym:
                continue
            candidate = sym + "USDT"
            # تحقق من وجود الزوج في Binance
            check = requests.get(f"{BINANCE_REST}/api/v3/exchangeInfo", params={"symbol": candidate}, timeout=10)
            if check.status_code == 200:
                symbols.append(candidate)
            if len(symbols) >= limit:
                break
        return symbols
    except Exception:
        return []

# --------------- Scanning logic -----------------
def analyze_symbol(symbol):
    try:
        df = fetch_binance_klines(symbol)
        close = df['close']
        macd, signal, hist = macd_series(close)
        rsi = rsi_series(close)
        latest_close = float(close.iloc[-1])
        latest_macd = float(macd.iloc[-1])
        latest_signal = float(signal.iloc[-1])
        latest_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None

        # شروط مبدئية ليتم اقتراح بيع (SHORT):
        # - MACD قطع هابط: macd < signal و قبلها كان macd > signal (cross down)
        # - RSI متجه نحو المرتفع (مثل > 60 أو >70) => إشارة تشبع شراء محتملة ثم انعكاس (للقصيرة نرغب بأن تكون مرتفعة)
        # - السعر يكون تحت مقاومات معينة (بسيط)
        macd_cross_down = False
        if len(macd) >= 3:
            if (macd.iloc[-3] > signal.iloc[-3]) and (macd.iloc[-1] < signal.iloc[-1]):
                macd_cross_down = True

        rsi_over = (latest_rsi is not None and latest_rsi >= 60)

        # Fibonacci levels (استخدم أقصى/أدنى من window  Fifty)
        lookback = min(50, len(df))
        recent_high, recent_low = get_swing_high_low(df['high'], lookback=lookback), get_swing_high_low(df['low'], lookback=lookback)
        # NOTE: get_swing_high_low returned tuple because of how defined; fix:
        recent_high = df['high'][-lookback:].max()
        recent_low = df['low'][-lookback:].min()
        fibs = fib_levels(recent_high, recent_low)

        # قرارات: لو التحققت الشروط السايقة نقترح صفقة بيع قصيرة
        should_short = macd_cross_down and rsi_over

        # إعداد سيناريو الصفقة
        if should_short:
            # entry: سعر السوق الحالي
            entry = latest_close
            # target1: 38.2% ، target2: 61.8% من Fibonacci
            target1 = fibs["38.2%"]
            target2 = fibs["61.8%"]
            # stoploss: فوق آخر Swing High بقليل
            stoploss = recent_high * 1.002  # +0.2% buffer
            # نسبة مخاطرة ممكن تتغير لاحقاً
            rr1 = (entry - target1) / (stoploss - entry) if (stoploss - entry) != 0 else None
            rr2 = (entry - target2) / (stoploss - entry) if (stoploss - entry) != 0 else None

            return {
                "symbol": symbol,
                "side": "SHORT",
                "entry": entry,
                "target1": target1,
                "target2": target2,
                "stoploss": stoploss,
                "macd": latest_macd,
                "signal": latest_signal,
                "rsi": latest_rsi,
                "fibs": fibs,
                "rr1": rr1,
                "rr2": rr2,
                "reason": "MACD cross down + RSI overbought"
            }
        else:
            return None
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        traceback.print_exc()
        return None

def compose_message(trades):
    if not trades:
        return f"لا توجد إشارات للبيع القصير (SHORT) الآن. (تاريخ: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})"
    lines = []
    lines.append(f"توصيات فيوتشر قصيرة - وقت المسح: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    for t in trades:
        lines.append(f"رمز: {t['symbol']}")
        lines.append(f"نوع الصفقة: {t['side']}")
        lines.append(f"سعر الدخول (سوق): {t['entry']:.6f}")
        lines.append(f"هدف 1: {t['target1']:.6f}  | هدف 2: {t['target2']:.6f}")
        lines.append(f"وقف خسارة: {t['stoploss']:.6f}")
        if t['rr1'] is not None:
            lines.append(f"Risk/Reward (entry→T1): {t['rr1']:.2f}   (entry→T2): {t['rr2']:.2f}")
        lines.append(f"MACD: {t['macd']:.6f}   Signal: {t['signal']:.6f}   RSI: {t['rsi']:.2f}")
        lines.append(f"نقطة موجبة: {t['reason']}")
        lines.append("----")
    return "\n".join(lines)

# --------------- Orchestration -----------------
def scan_and_alert(chat_id):
    try:
        symbols = get_symbols_from_cmc(SYMBOL_LIMIT) or get_top_symbols_from_binance(SYMBOL_LIMIT)
        trades = []
        for symbol in symbols:
            # نتأكد أنه زوج موجود وصالح لقراءة الشموع
            try:
                res = analyze_symbol(symbol)
                if res:
                    trades.append(res)
            except Exception as e:
                print(f"error analyzing {symbol}: {e}")
        msg = compose_message(trades)
        if chat_id:
            bot.send_message(chat_id=chat_id, text=msg)
        else:
            print("No chat_id provided and no user to send. Message:\n", msg)
    except Exception as e:
        print("Scan failed:", e)
        traceback.print_exc()

def background_loop(get_chat_id_callable):
    while True:
        try:
            chat_id = get_chat_id_callable()
            print(f"[{datetime.utcnow()}] Starting scan for chat_id={chat_id}")
            scan_and_alert(chat_id)
        except Exception as e:
            print("background error:", e)
            traceback.print_exc()
        time.sleep(SLEEP_SECONDS)

# --------------- Telegram command handlers -------------
def start_handler(update: Update, context: CallbackContext):
    global CHAT_ID
    user = update.effective_user
    CHAT_ID = update.effective_chat.id
    # لو المستخدم يريد يستخدم البوت الخاص به كنقطة استقبال
    update.message.reply_text(f"تم تسجيل الشات ID = {CHAT_ID}. سأنفذ المسح كل {INTERVAL_MIN} دقيقة وأرسل التوصيات هنا.")
    print(f"Registered CHAT_ID = {CHAT_ID}")

def scan_now_handler(update: Update, context: CallbackContext):
    cid = update.effective_chat.id
    update.message.reply_text("جارٍ تنفيذ فحص الآن...")
    try:
        scan_and_alert(cid)
    except Exception as e:
        update.message.reply_text(f"فشل الفحص: {e}")

def help_handler(update: Update, context: CallbackContext):
    update.message.reply_text("/start - تسجيل الشات الحالي\n/scan - تشغيل فحص يدوي الآن\n/help - المساعدة")

# --------------- Main --------------------------------
def get_chat_id():
    # prefer ENV CHAT_ID, else registered via /start
    return CHAT_ID

def main():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CommandHandler("scan", scan_now_handler))
    dp.add_handler(CommandHandler("help", help_handler))

    # start polling for commands
    updater.start_polling()

    # start background scanner thread
    t = threading.Thread(target=background_loop, args=(get_chat_id,), daemon=True)
    t.start()

    print("Bot started. Waiting for commands and running background scanner.")
    updater.idle()

if __name__ == "__main__":
    main()
