Python 3.12.4

В папке wine_app:
python -m pip install -r requirements.txt
python -m streamlit run app.py

открыть http://localhost:8501 в браузере

Ссылка на Streamlit Community Cloud: https://xwqj6otrfab69hjhzh77pa.streamlit.app/

4.5:

@st.cache_resource хранит один экземпляр модели. Без joblib.load модель загружалась бы  с диска при каждом движении ползунка.

@st.cache_data хранит таблицу и рассчитанные вероятности, возвращая их копии. При смене порога обновляются метрики, а вероятности берутся из кэша.
