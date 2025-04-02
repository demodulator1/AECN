import pandas as pd
# 设置 pandas 的显示选项
# 设置最大列数为 None，表示不限制显示的列数
pd.set_option('display.max_columns', None)
# 设置最大列宽为 None，表示不限制列的显示宽度
pd.set_option('display.max_colwidth', None)
def read_data():
    df_1580 = pd.read_excel('query-hive-1580_processed.xlsx')
    df_1579 = pd.read_excel('query-hive-1579_processed.xlsx')
    # print(df_1580.head())
    # print(df_1579.head())
    
    # 提取并重组列
    df_1579_selected=df_1579.iloc[:,[2,8,3,9,4,10]]
    df_1580_selected=df_1580.iloc[:,[1,7,2,8,3,9]]
    # print(df_1579_selected.head())
    # print(df_1580_selected.head())

    return df_1579_selected.values.tolist(), df_1580_selected.values.tolist()