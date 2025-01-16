import pandas as pd
from pathlib import Path

current_file_path = Path(__file__).resolve()
parent_directory = current_file_path.parent


def load_data():
    return pd.read_csv("../data/raw/orders_autumn_2020.csv")


load_data()
