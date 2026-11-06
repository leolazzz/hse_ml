
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(BASE / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(BASE / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
df = load_test()
X = df.drop(columns='target')
y = df['target']
p = test_proba(model, X)

config = json.loads((BASE / 'config.json').read_text())
threshold = st.sidebar.slider(
    'Порог', 0.0, 1.0, float(config['threshold']), 0.001
)

st.title('Подбор хороших вин')

tab1, tab2, tab3, tab4 = st.tabs([
    'Одно вино',
    'Модельные метрики',
    'Загрузить CSV',
    'Коэффициенты'
])


# 4.1
with tab1:
    wine = {}

    for name in X.columns:
        if name == 'is_red':
            wine[name] = int(st.toggle(
                'Красное вино (выкл. белое)',
                value=bool(X[name].median())
            ))
        else:
            wine[name] = st.number_input(
                name,
                min_value=float(X[name].min()),
                max_value=float(X[name].max()),
                value=float(X[name].median()),
                format='%.4f'
            )

    row = pd.DataFrame([wine], columns=X.columns)
    chance = model.predict_proba(row)[0, 1]

    st.write(f'Вероятность хорошего вина: {chance:.3f}')
    st.write(
        'Рекомендовать'
        if chance >= threshold
        else 'Не рекомендовать'
    )


# 4.2
with tab2:
    pred = (p >= threshold).astype(int)

    precision = metrics.precision_score(
        y, pred, zero_division=0
    )
    recall = metrics.recall_score(y, pred)
    f1 = metrics.f1_score(y, pred, zero_division=0)

    cm = metrics.confusion_matrix(y, pred)

    st.write('Матрица ошибок:')
    st.dataframe(pd.DataFrame(
        cm,
        index=['Факт 0', 'Факт 1'],
        columns=['Прогноз 0', 'Прогноз 1']
    ))

    st.write(f'Precision: {precision:.3f}')
    st.write(f'Recall: {recall:.3f}')
    st.write(f'F1: {f1:.3f}')

    fpr, tpr, _ = metrics.roc_curve(y, p)
    fpr_point = cm[0, 1] / (cm[0, 0] + cm[0, 1])

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.scatter(fpr_point, recall, color='red')
    ax.set(
        xlabel='FPR',
        ylabel='TPR',
        title='ROC'
    )
    st.pyplot(fig)
    plt.close(fig)

    prec_curve, rec_curve, _ = (
        metrics.precision_recall_curve(y, p)
    )

    fig, ax = plt.subplots()
    ax.plot(rec_curve, prec_curve)
    ax.scatter(recall, precision, color='red')
    ax.set(
        xlabel='Recall',
        ylabel='Precision',
        title='PR'
    )
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'Если увеличить порог, в подборке обычно будет '
        'меньше вин: precision растёт, recall падает. '
        'При уменьшении порога наоборот.'
    )


# 4.3
with tab3:
    uploaded = st.file_uploader(
        'Загрузите CSV с признаками вин',
        type='csv'
    )

    if uploaded is not None:
        new = pd.read_csv(uploaded)
        missing = X.columns.difference(new.columns)

        if len(missing):
            st.error(
                'Не хватает столбцов: ' + ', '.join(missing)
            )
        else:
            result = new.copy()
            result['probability'] = model.predict_proba(
                new[X.columns]
            )[:, 1]
            result['prediction'] = (
                result['probability'] >= threshold
            ).astype(int)

            st.dataframe(result)

            st.download_button(
                'Скачать результаты',
                result.to_csv(index=False),
                file_name='predictions.csv',
                mime='text/csv'
            )


# 4.4
with tab4:
    coefs = pd.Series(
        model.named_steps['logisticregression'].coef_[0],
        index=X.columns
    ).sort_values()

    fig, ax = plt.subplots()
    coefs.plot.barh(ax=ax)
    ax.set(
        xlabel='Коэффициент после масштабирования',
        title='Влияние признаков'
    )
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'Сильнее всего за хорошее вино:',
        coefs.idxmax()
    )
    st.write(
        'Сильнее всего против хорошего вина:',
        coefs.idxmin()
    )