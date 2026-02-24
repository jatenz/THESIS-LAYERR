import sqlite3
import pandas as pd

def write_df(db_path: str, table: str, df: pd.DataFrame):
    con = sqlite3.connect(db_path)
    df.to_sql(table, con, if_exists="replace", index=False)
    con.close()
