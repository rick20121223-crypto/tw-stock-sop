"""
台灣時區(UTC+8)輔助函式。

GitHub Actions runner是UTC時間，Python的date.today()/datetime.now()在
runner上也是UTC。各個workflow的cron雖然已經把UTC/台灣時間的offset算過、
讓「觸發時間點」對上台灣的某個時刻（例如notify.yml註解寫的「UTC前一天
22:00 = 台灣當天06:00」），但那只保證「什麼時候被觸發」，程式內部只要
還用date.today()/datetime.now()判斷「今天日期」「今天星期幾」，得到的
還是UTC的日期——在台灣時間清晨00:00~08:00執行時，UTC日期會是台灣的
前一天，拿去跟同一天稍晚（或另一支腳本）用台灣時間認知寫入的資料比對，
就會出現日期對不上的bug（實際發生過：notify_email.py的_already_ran_
today() 因此把整天的批次誤判成「已經跑過」而跳過；institutional_
ranking.py的rotate指令因此把「上週」算成早了一週）。

這個模組被 notify_email.py、notify_intraday.py、institutional_ranking.py
共用，任何新腳本要判斷「今天」都應該從這裡拿，不要各自直接呼叫
date.today()/datetime.now()。
"""

from datetime import date, datetime, timedelta, timezone

TAIWAN_TZ = timezone(timedelta(hours=8))


def taiwan_now() -> datetime:
    """回傳目前的台灣時間（帶時區資訊）。"""
    return datetime.now(TAIWAN_TZ)


def taiwan_today() -> date:
    """回傳台灣時區的今天日期，不受執行環境(如GitHub Actions runner)本身時區影響。"""
    return taiwan_now().date()
