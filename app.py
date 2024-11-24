import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sb
import io

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

class ConnectionDatabase:
    def __init__(self):
        self.DB_CONNECT_DONALDSON = 'postgresql://neondb_owner:q52WaZEsiQkd@ep-aged-band-a5ft7yvk.us-east-2.aws.neon.tech/neondb?sslmode=require'
        self.DB_CONNECT_STAL = 'postgresql://neondb_owner:trT4F3ORExaS@ep-fancy-wind-a5m7rl2y.us-east-2.aws.neon.tech/neondb?sslmode=require'

    def connect_db(self, fish_type):
        if fish_type.lower() == "stal":
            DB_CONNECT = self.DB_CONNECT_STAL
        elif fish_type.lower() == "donaldson":
            DB_CONNECT = self.DB_CONNECT_DONALDSON
        engine = create_engine(DB_CONNECT)
        return engine

def save_data_to_db(fish_type, form_data):
    engine = ConnectionDatabase().connect_db(fish_type)
    with engine.begin() as connection:
        data_df = pd.DataFrame([form_data])
        data_df.to_sql('data', con=connection, if_exists='append', index=False)

def show_form(fish_type):
    st.header(f"Введите данные для {fish_type.capitalize()}")

    columns = [
        '№ самки', 'вес тела, г', 'Длина по Смитту, см', 'Высота тела, см',
        'толщина тела, см', 'вес икры, г', 'вес икринки, мг', 'К упит', 'И толщ',
        'И выс', 'Доля икры, %', 'рабочая плодовитость, тыс. шт', 'относительная плодовитость, шт/кг ',
        'индекс репродуктивности, г/кг', 'вес икры заложенной, г', 'заложено икры, тыс. шт*',
        'погибшая икра до глазка, шт', 'погибшая икра после глазка, шт', 'неоплодотворенная икра, шт',
        '% гибели до глазка', '% гибели после глазка', '% опл', '% выживаемости', '% выклева',
        'свободные эмбрионы, тыс. шт.', 'примечание'
    ]

    form_data = {}

    for column in columns:
        form_data[column] = st.text_input(f"Введите {column}: ")

    if st.button('Сохранить данные'):
        save_data_to_db(fish_type, form_data)
        st.success(f"Данные для {fish_type} успешно сохранены!")

def main():
    st.title("Ввод данных о рыбе")

    fish_type = st.radio("Выберите тип рыбы", ["Stal (Сталеголовый лосось)", "Donaldson (Дональдсон)"])

    fish_type = "stal" if "stal" in fish_type.lower() else "donaldson"

    show_form(fish_type)

    if st.button("Перейти к анализу данных"):
        st.write("Переход на сайт с анализом данных...")

if __name__ == "__main__":
    main()


col1, col2 = st.columns([1, 5])

with col1:
    st.image("Leonardo_Phoenix_Render_a_detailed_colorful_illustration_of_a_3.jpg", width=90)

with col2:
    st.markdown("<h1>Анализ данных из файла</h1>", unsafe_allow_html=True)

fish_type = st.selectbox("Выберите тип рыбы", ["Stal (Сталеголовый лосось)", "Donaldson (Дональдсон)"])

class BonitisationConfig:
    CORR_COLUMNS =  [
        'вес тела, г',
        'Длина по Смитту, см',
        'Высота тела, см',
        'толщина тела, см',
        'вес икры, г',
        'вес икринки, мг',
        'К упит', 'И толщ',
        'И выс', 'Доля икры, %',
        'рабочая плодовитость, тыс. шт',
        'относительная плодовитость, шт/кг ',
        'вес икры заложенной, г'
    ]

class ConnectionDatabase:
    def __init__(self):
        self.DB_CONNECT_DONALDSON = 'postgresql://neondb_owner:q52WaZEsiQkd@ep-aged-band-a5ft7yvk.us-east-2.aws.neon.tech/neondb?sslmode=require'
        self.DB_CONNECT_STAL = 'postgresql://neondb_owner:trT4F3ORExaS@ep-fancy-wind-a5m7rl2y.us-east-2.aws.neon.tech/neondb?sslmode=require'

    def connect_db(self, fish_type):
        if fish_type.lower() == "stal":
            DB_CONNECT = self.DB_CONNECT_STAL
        elif fish_type.lower() == "donaldson":
            DB_CONNECT = self.DB_CONNECT_DONALDSON
        engine = create_engine(DB_CONNECT)
        with engine.begin() as connection:
            table = pd.read_sql_table('data', con=connection)
        return table

class Bonitisation:
    def __init__(self, data):
        self.data = data
        self.corr_columns = BonitisationConfig.CORR_COLUMNS

    def get_corr_matrix(self, show=True):
        missing_columns = [col for col in self.corr_columns if col not in self.data.columns]
        if missing_columns:
            return f"Отсутствуют столбцы: {missing_columns}"

        corr_matrix = self.data[self.corr_columns].corr()
        if show:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            fig, ax = plt.subplots(figsize=(10, 8))
            sb.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", mask=mask, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        return corr_matrix

import pandas as pd
import numpy as np

class GetStatistics:
    def __init__(self, data):
        self.data = data
        self.ROUND_VALUE = 3

    def get_stats(self):
        numeric_columns = self.data.select_dtypes(include=[np.number])
        results = {
            "Column": [],
            "Mean": [],
            "Max": [],
            "Min": [],
            "Sigma (Std Dev)": [],
            "Error X": [],
            "CV (%)": [],
            "X+1.5δ": [],
            "X-1δ": [],
            "X-0.5δ": [],
        }

        for column in numeric_columns.columns:
            col_data = round(numeric_columns[column], self.ROUND_VALUE)
            mean = round(col_data.mean(), self.ROUND_VALUE)
            max_val = round(col_data.max(), self.ROUND_VALUE)
            min_val = round(col_data.min(), self.ROUND_VALUE)
            sigma = round(col_data.std(), self.ROUND_VALUE)
            error_x = round(sigma / np.sqrt(len(col_data)), self.ROUND_VALUE)
            cv = round((sigma / mean) * 100 if mean != 0 else 0, self.ROUND_VALUE)
            x_plus_1_5d = round(mean + 1.5 * sigma, self.ROUND_VALUE)
            x_minus_1d = round(mean - sigma, self.ROUND_VALUE)
            x_minus_0_5d = round(mean - 0.5 * sigma, self.ROUND_VALUE)

            results["Column"].append(column)
            results["Mean"].append(mean)
            results["Max"].append(max_val)
            results["Min"].append(min_val)
            results["Sigma (Std Dev)"].append(sigma)
            results["Error X"].append(error_x)
            results["CV (%)"].append(cv)
            results["X+1.5δ"].append(x_plus_1_5d)
            results["X-1δ"].append(x_minus_1d)
            results["X-0.5δ"].append(x_minus_0_5d)

        results_df = pd.DataFrame(results)
        return results_df

    def classify_fish(self, stats_df):
        classification_data = []

        for _, row in stats_df.iterrows():
            column = row["Column"]
            mean = row["Mean"]
            sigma = row["Sigma (Std Dev)"]
            x_minus_0_5d = row["X-0.5δ"]
            x_minus_1_5d = row["X-1δ"]

            classification_result = []
            for fish_value in self.data[column]:
                if fish_value >= x_minus_0_5d:
                    classification_result.append("Elite")
                elif x_minus_1_5d <= fish_value < x_minus_0_5d:
                    classification_result.append("Class 1")
                else:
                    classification_result.append("Class 2")

            classification_data.append({
                "Column": column,
                "Classification": classification_result
            })


        classification_df = pd.DataFrame({
            "Fish ID": self.data.index
        })
        for col_data in classification_data:
            classification_df[col_data["Column"]] = col_data["Classification"]

        return classification_df

    
def draw_histograms(data, num_rows=3, num_cols=4):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    column_count = len(numeric_columns)
    index = 0
    histogram_images = []

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_index in range(num_cols):
            if index < column_count:
                with cols[col_index]:
                    column_name = numeric_columns[index]
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(data[column_name].dropna(), bins=20, color='skyblue', edgecolor='black')
                    ax.set_title(f"Histogram: {column_name}")
                    ax.set_xlabel(column_name)
                    ax.set_ylabel("Frequency")
                    img_buf = io.BytesIO()
                    st.pyplot(fig)
                    plt.close(fig)
                    img_buf.seek(0)
                    histogram_images.append((column_name, img_buf))
                index += 1

    return histogram_images


st.title("Анализ данных и критерия элитности рыбы")

if fish_type.startswith("Stal"):
    data = ConnectionDatabase().connect_db("stal")
    st.subheader("Сталеголовый лосось")
else:
    data = ConnectionDatabase().connect_db("donaldson")
    st.subheader("Дональдсон")

st.write("Предварительный просмотр данных:")
st.write(data.head())

stats = GetStatistics(data).get_stats()
st.subheader("Основные статистики")
st.write(stats)

classified_stats = GetStatistics(data).classify_fish(stats)

st.subheader("Классификация рыбы по элитности")
st.write(classified_stats.iloc[:, 3:])

st.subheader("Рейтинг рыб")
st.write(classified_stats.iloc[:, 3:].apply(pd.Series.value_counts, axis=1).fillna(0).astype(int))

csv_classification = classified_stats.to_csv(index=False)
st.download_button(label="Скачать таблицу классификации (CSV)", data=csv_classification, file_name="fish_classification.csv", mime="text/csv")

st.subheader("Матрица корреляции")
bonitisation = Bonitisation(data)
corr_matrix = bonitisation.get_corr_matrix(show=True)

csv_corr = corr_matrix.to_csv(index=True)
st.download_button(label="Скачать матрицу корреляции (CSV)", data=csv_corr, file_name="correlation_matrix.csv", mime="text/csv")

st.subheader("Гистограммы")
histograms = draw_histograms(data, num_rows=3, num_cols=3)

for column_name, img_buf in histograms:
    st.download_button(label=f"Скачать гистограмму для {column_name}", data=img_buf, file_name=f"{column_name}_histogram.png", mime="image/png")
					 
