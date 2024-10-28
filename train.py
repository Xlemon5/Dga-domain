import pandas as pd
import random
import string
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def generate_random_string(length, use_digits=True):
    """
    Генерирует случайную строку заданной длины, включающую буквы и опционально цифры.

    :param length: Длина строки
    :param use_digits: Включать ли цифры в строку
    :return: Случайная строка
    """
    characters = string.ascii_lowercase
    if use_digits:
        characters += string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def generate_domain_names(count):
    """
    Генерирует список доменных имён с различными паттернами и TLD.

    :param count: Количество доменных имён для генерации
    :return: Список сгенерированных доменных имён
    """
    tlds = ['.com', '.ru', '.net', '.org', '.de', '.edu', '.gov', '.io', '.shop', '.co', '.nl', '.fr', '.space', '.online', '.top', '.info']

    def generate_domain_name():
        tld = random.choice(tlds)
        patterns = [
            lambda: generate_random_string(random.randint(5, 10), use_digits=False) + '-' + generate_random_string(random.randint(5, 10), use_digits=False),
            lambda: generate_random_string(random.randint(8, 12), use_digits=False),
            lambda: generate_random_string(random.randint(5, 7), use_digits=False) + '-' + generate_random_string(random.randint(2, 4), use_digits=True),
            lambda: generate_random_string(random.randint(4, 6), use_digits=False) + generate_random_string(random.randint(3, 5), use_digits=False),
            lambda: generate_random_string(random.randint(3, 5), use_digits=False) + '-' + generate_random_string(random.randint(3, 5), use_digits=False),
        ]
        domain_pattern = random.choice(patterns)
        return domain_pattern() + tld

    domain_list = [generate_domain_name() for _ in range(count)]
    return domain_list

def n_grams(domain):
    """
    Генерирует n-граммы для доменного имени.

    :param domain: Доменное имя
    :return: Список n-грамм
    """
    grams_list = []
    # Размеры n-грамм
    n = [3, 4, 5]
    domain = domain.split('.')[0]
    for count_n in n:
        for i in range(len(domain)):
            if len(domain[i: count_n + i]) == count_n:
                grams_list.append(domain[i: count_n + i])
    return grams_list

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        logging.info('Загрузка данных')
        part_df = pd.read_csv('top-1m.csv')
        df_val = pd.read_csv('val_df.csv')
        df_test = pd.read_csv('test_df.csv')
        logging.info('Данные успешно загружены.')
    except Exception as e:
        logging.error(f'Ошибка при загрузке данных: {e}')

    logging.info('Обработка данных')
    part_df = part_df.drop('1', axis=1)
    part_df.rename(columns={'google.com': 'domain'}, inplace=True)
    part_df['is_dga'] = 0
    list_dga = df_val[df_val.is_dga == 1].domain.tolist()
    generated_domains = generate_domain_names(1000000)
    part_df_dga = pd.DataFrame({
        'domain': generated_domains,
        'is_dga': [1] * len(generated_domains)
    })
    df = pd.concat([part_df, part_df_dga], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # Исключение доменов из валидационного и тестового наборов
    train_set = set(df.domain.tolist())
    val_set = set(df_val.domain.tolist())
    test_set = set(df_test.domain.tolist())
    intersection_val = train_set.intersection(val_set)
    intersection_test = train_set.intersection(test_set)
    if intersection_val or intersection_test:
        df = df[~df['domain'].isin(intersection_val | intersection_test)]


    # Балансировка классов до одинакового числа примеров
    logging.info('Балансировка классов')
    df_train_0 = df[df['is_dga'] == 0]
    df_train_1 = df[df['is_dga'] == 1]
    num_samples_per_class = 500000
    df_train_0_sampled = df_train_0.sample(n=num_samples_per_class, random_state=42)
    df_train_1_sampled = df_train_1.sample(n=num_samples_per_class, random_state=42)
    df_balanced = pd.concat([df_train_0_sampled, df_train_1_sampled])
    df_train = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info('Создание и обучение модели')
    # Модель LogisticRegression выбрана, так как при обучении CatBoost не хватало памяти
    model_pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(tokenizer=n_grams, token_pattern=None)),
        ("model", LogisticRegression(solver='saga', n_jobs=-1, random_state=12345))
    ])

    model_pipeline.fit(df_train['domain'], df_train['is_dga'])
    logging.info('Сохранение модели')
    joblib_file = "model_pipeline.pkl"
    joblib.dump(model_pipeline, joblib_file)
    logging.info(f'Модель сохранена в {joblib_file}')