"""
App 進入點（st.navigation 路由）。

依使用者指示，導覽列精簡成只留 3 頁：長線留倉／短線進場／多週期整合
分析。「日線快篩」「個股詳細分析」程式碼都還在（見 pages/_2_日線快篩.py、
pages/_4_個股詳細分析.py，檔名開頭底線只是標記「暫不在導覽列」，跟
Streamlit本身的行為無關），只是沒有放進下面 pages 清單，不會顯示在
導覽列、也不能透過網址直接開啟（st.navigation 一旦啟用，就會接管全部
導覽、忽略 pages/ 資料夾自動掃描這件事本身）。之後想恢復顯示，把對應
的 st.Page(...) 加回下面的清單即可。

執行方式：
    streamlit run streamlit_app.py
"""

import streamlit as st

pages = [
    st.Page("home.py", title="長線留倉", icon="📅", default=True),
    st.Page("pages/1_短線進場.py", title="短線進場", icon="⚡"),
    st.Page("pages/3_多週期整合分析.py", title="多週期整合分析", icon="🔍"),
]

pg = st.navigation(pages)
pg.run()
