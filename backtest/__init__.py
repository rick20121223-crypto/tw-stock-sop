"""指標搭配回測模組：獨立於主站判讀邏輯（不匯入/不修改 sop_decision.py、sop_rules.yaml），
只借用 stock_core.py 既有的指標計算函式（MACD/KD/BBI/MA），避免重算一份不一致的版本。
只輸出 CSV 與 markdown 報告，不碰 Streamlit 頁面。"""
