import pandas as pd


def extract_average_ratings(rating_file, prefixes, rating_columns):
    df = pd.read_excel(rating_file).iloc[1:]
    df['Key'] = df.iloc[:, 0].apply(lambda x: x.split('some_postfix')[0])

    avg_ratings = {}
    for prefix, column in zip(prefixes, rating_columns):
        rating_dict = df.groupby('Key')[column].mean().to_dict()
        avg_ratings.update({f"{prefix}_{k}": v for k, v in rating_dict.items()})
    return avg_ratings
