import joblib
import pandas as pd
import logging
from utils import n_grams
from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


def get_metrics(df):
    """
    Рассчитывает метрики производительности модели и записывает их в файл.

    :param df: DataFrame с истинными метками и предсказаниями модели
    :return: None
    """
    y_true = df['is_dga']
    y_pred = df['predict']

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        with open('validation.txt', 'w') as f:
            f.write(f'True Positive (TP): {tp}\n')
            f.write(f'False Positive (FP): {fp}\n')
            f.write(f'False Negative (FN): {fn}\n')
            f.write(f'True Negative (TN): {tn}\n')
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'Precision: {precision:.4f}\n')
            f.write(f'Recall: {recall:.4f}\n')
            f.write(f'F1 Score: {f1:.4f}\n')
        logging.info('Метрики успешно записаны в файл validation.txt.')
    except IOError as e:
        logging.error(f'Ошибка при записи метрик в файл: {e}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        logging.info('Загрузка данных')
        df_val = pd.read_csv('val_df.csv')
        logging.info('Данные успешно загружены.')
    except Exception as e:
        logging.error(f'Ошибка при загрузке данных: {e}')
        exit(1)

    try:
        logging.info('Загрузка модели')
        model_pipeline = joblib.load("model_pipeline.pkl")
        logging.info('Модель успешно загружена.')
    except Exception as e:
        logging.error(f'Ошибка при загрузке модели: {e}')
        exit(1)

    df_val['predict'] = model_pipeline.predict(df_val.domain)
    logging.info('Предсказания успешно сделаны.')

    get_metrics(df_val)