# import pandas as pd
# import json
# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
# import matplotlib.pyplot as plt
# import glob
# from pylab import mpl
# import os
# from tqdm import tqdm
# import time

# plt.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False
# # 1. Load Parquet data


# def load_parquet_data(parquet_path):
#     return pd.read_parquet(parquet_path)

# def load_product_catalog(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         catalog = json.load(f)
#     products = pd.DataFrame(catalog.get('products', []))
#     return products.rename(columns={'id': 'item_id'})

# # 2. Explode purchase_history

# def explode_purchase_history(df, products_df):
#     records = []
#     for _, row in tqdm(df.iterrows()):
#         history = row['purchase_history']
#         history = json.loads(history)
#         for item in history.get('items'):
#             item_ids = item.get('id')
            
#             if not isinstance(item_ids, list):
#                 item_ids = [item_ids]

#             for item_id in item_ids:
#                 prod = products_df[products_df['item_id'] == item_id]
#                 if prod.empty:
#                     continue
#                 prod = prod.iloc[0]
#                 record = {
#                     'order_id': row.get('order_id'),
#                     'item_id': item_id,
#                     'item_count': 1,
#                     'price': prod['price'],
#                     'category': prod['category'],
#                     'payment_status': history.get('payment_status'),
#                     'payment_method': history.get('payment_method'),
#                     'purchase_date': pd.to_datetime(history.get('purchase_date'))
#                 }
#                 records.append(record)
#     return pd.DataFrame(records)

# # 3. Association rule mining helpers

# def mine_rules(transactions, min_support, min_confidence, algorithm='apriori'):
#     te = TransactionEncoder()
#     te_ary = te.fit(transactions).transform(transactions)
#     df_te = pd.DataFrame(te_ary, columns=te.columns_)
#     if algorithm == 'apriori':
#         freq_itemsets = apriori(df_te, min_support=min_support, use_colnames=True)
#     else:
#         freq_itemsets = fpgrowth(df_te, min_support=min_support, use_colnames=True)
#     rules = association_rules(freq_itemsets, metric='confidence', min_threshold=min_confidence)
#     return freq_itemsets, rules

# # 3.1 商品类别关联规则

# def category_association(df, min_support=0.02, min_confidence=0.5, algorithm='apr'):
#     orders = df.groupby('order_id')['category'].apply(list).tolist()
#     algo = 'apriori' if algorithm.startswith('apr') else 'fpgrowth'
#     freq, rules = mine_rules(orders, min_support, min_confidence, algorithm=algo)
#     rules_elec = rules[
#         rules['antecedents'].apply(lambda x: any('电子' in str(i) for i in x)) |
#         rules['consequents'].apply(lambda x: any('电子' in str(i) for i in x))
#     ]
#     return freq, rules, rules_elec

# # 3.2 支付方式与商品类别关联

# def payment_category_association(df, min_support=0.01, min_confidence=0.6, algorithm='apr'):
#     df['high_value'] = df['price'] > 5000
#     trans_by_method = df.groupby('payment_method')['category'].apply(list).tolist()
#     algo = 'apriori' if algorithm.startswith('apr') else 'fpgrowth'
#     freq, rules = mine_rules(trans_by_method, min_support, min_confidence, algorithm=algo)
#     high_methods = df[df['high_value']]['payment_method'].value_counts()
#     return freq, rules, high_methods

# # 4. 时间序列模式

# def time_series_patterns(df):
#     df['quarter'] = df['purchase_date'].dt.to_period('Q')
#     df['month'] = df['purchase_date'].dt.to_period('M')
#     df['weekday'] = df['purchase_date'].dt.day_name()

#     season_counts = df.groupby(['quarter', 'category']).size().unstack(fill_value=0)
#     month_counts = df.groupby(['month', 'category']).size().unstack(fill_value=0)
#     weekday_counts = df.groupby(['weekday', 'category']).size().unstack(fill_value=0)

#     df_sorted = df.sort_values(['order_id', 'purchase_date'])
#     seqs = []
#     for _, grp in df_sorted.groupby('order_id'):
#         cats = grp['category'].tolist()
#         for i in range(len(cats) - 1):
#             seqs.append((cats[i], cats[i+1]))
#     seq_df = pd.DataFrame(seqs, columns=['A', 'B'])
#     seq_counts = seq_df.value_counts().reset_index(name='count')
#     return season_counts, month_counts, weekday_counts, seq_counts

# # 5. 退款模式

# def refund_pattern_analysis(df, min_support=0.005, min_confidence=0.4, algorithm='apr'):
#     refunded = df[df['payment_status'].isin(['已退款', '部分退款'])]
#     orders = refunded.groupby('order_id')['category'].apply(list).tolist()
#     algo = 'apriori' if algorithm.startswith('apr') else 'fpgrowth'
#     freq, rules = mine_rules(orders, min_support, min_confidence, algorithm=algo)
#     return freq, rules

# # 6. 可视化示例

# def plot_seasonality(season_counts):
#     season_counts.plot(kind='bar', figsize=(10, 6))
#     plt.title('季度购买频率')
#     plt.xlabel('季度')
#     plt.ylabel('购买次数')
#     plt.tight_layout()
#     plt.show()

# # 7. 主流程

# def main(parquet_path, products_json_path):
#     start_time = time.time()
#     df_raw = load_parquet_data(parquet_path)
#     products_df = load_product_catalog(products_json_path)
#     print(len(df_raw))
#     df = explode_purchase_history(df_raw, products_df)
#     print("#1")
#     fi_cat, rules_cat, rules_elec = category_association(df)
#     print('类别频繁项集:', fi_cat)
#     print('类别关联规则:', rules_cat)
#     print('电子产品相关规则:', rules_elec)
#     print("#2")
#     fi_pay, rules_pay, high_methods = payment_category_association(df)
#     print('支付-类别频繁项集:', fi_pay)
#     print('支付关联规则:', rules_pay)
#     print('高价值商品首选支付方式:', high_methods)
#     print("#3")
#     season, month, weekday, seq = time_series_patterns(df)
#     plot_seasonality(season)
#     print('季度模式:', season)
#     print('序列 A->B 前几条:', seq.head())
#     print("#4")
#     fi_refund, rules_refund = refund_pattern_analysis(df)
#     print('退款模式频繁项集:', fi_refund)
#     print('退款关联规则:', rules_refund)
#     end_time = time.time()
#     print(f"\n数据统计和处理耗时: {end_time - start_time:.2f} 秒")

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='购物历史关联规则挖掘')
#     parser.add_argument('--input_parquet', required=False, default="/home/hfxia/data_trans/30G_data_new/part-00000.parquet")
#     parser.add_argument('--products_json', required=False , default="/home/hfxia/data_trans/product_catalog.json")
#     args = parser.parse_args()
#     main(args.input_parquet, args.products_json)
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from scipy.sparse import csr_matrix

# Matplotlib 中文配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
def load_parquet_data(parquet_path):
    return pd.read_parquet(parquet_path)

def load_product_catalog(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    products = pd.DataFrame(catalog.get('products', []))
    return products.rename(columns={'id': 'item_id'})

# 2. 向量化 Explode + Merge
def explode_purchase_history(df_raw, products_df):
    df = df_raw.copy()

    # 2.1 给每条记录分配唯一 order_id
    df['order_id'] = range(len(df))

    # 2.2 解析 JSON 字符串为 dict
    df['history_obj'] = df['purchase_history'].apply(json.loads)

    # 2.3 取出 “items” 列，展开成多行
    df['items_list'] = df['history_obj'].apply(lambda h: h.get('items', []))
    df = df.explode('items_list').dropna(subset=['items_list'])

    # 2.4 items_list 里取出 item_id（可能是单值或列表），再 explode
    df['item_id'] = df['items_list'].apply(lambda x: x.get('id'))
    df = df.explode('item_id').dropna(subset=['item_id'])

    # 2.5 与产品目录一次性 merge
    df = df.merge(products_df, on='item_id', how='inner')

    # 2.6 从 history_obj 中抽出其它字段
    df['payment_status']  = df['history_obj'].apply(lambda h: h.get('payment_status'))
    df['payment_method']  = df['history_obj'].apply(lambda h: h.get('payment_method'))
    df['purchase_date']   = pd.to_datetime(df['history_obj'].apply(lambda h: h.get('purchase_date')))
    df['item_count']      = 1

    # 2.7 只保留需要的列
    return df[[
        'order_id',
        'item_id',
        'item_count',
        'price',
        'category',
        'payment_status',
        'payment_method',
        'purchase_date'
    ]]

# 3. 关联规则挖掘通用函数（带多重检查与回退）
def mine_rules(transactions, min_support, min_confidence, algorithm='apriori'):
    # 3.1 过滤空交易并统一转 str
    trans = [list(map(str, t)) for t in transactions if isinstance(t, (list, tuple)) and t]
    if not trans:
        return pd.DataFrame(), pd.DataFrame()

    te = TransactionEncoder()
    te.fit(trans)
    # 如果没有任何项，则直接返回空
    if not getattr(te, 'columns_', None):
        return pd.DataFrame(), pd.DataFrame()

    # 3.2 尝试用稀疏矩阵
    
    try:
        sparse_ary = te.transform(trans, sparse=True)
        sparse_csr = csr_matrix(sparse_ary)
        df_te = pd.DataFrame.sparse.from_spmatrix(sparse_csr, columns=te.columns_)
    except Exception:
        # 回退到密集矩阵
        dense = te.transform(trans, sparse=False)
        df_te = pd.DataFrame(dense, columns=te.columns_)
    
    # 如果没有任何列，也直接返回空
    if df_te.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame()

    # 3.3 挖掘频繁项集 & 关联规则
    if algorithm == 'apriori':
        freq_itemsets = apriori(df_te, min_support=min_support, use_colnames=True, low_memory=True)
    else:
        freq_itemsets = fpgrowth(df_te, min_support=min_support, use_colnames=True)
    
    if freq_itemsets.empty:
        return freq_itemsets, pd.DataFrame()
    
    rules = association_rules(freq_itemsets, metric='confidence', min_threshold=min_confidence)
    return freq_itemsets, rules

# 3.1 商品类别关联规则
def category_association(df, min_support=0.02, min_confidence=0.5, algorithm='apr'):
    orders = df.groupby('order_id')['category'].apply(list).tolist()
    algo = 'apriori' if algorithm.startswith('apr') else 'fpgrowth'

    freq, rules = mine_rules(orders, min_support, min_confidence, algo)
    
    rules_elec = pd.DataFrame()
    if not rules.empty:
        print("sadasda")
        rules_elec = rules[
            rules['antecedents'].apply(lambda x: any('电子' in str(i) for i in x)) |
            rules['consequents'].apply(lambda x: any('电子' in str(i) for i in x))
        ]
    return freq, rules, rules_elec

# 3.2 支付方式与商品类别关联
def payment_category_association(df, min_support=0.01, min_confidence=0.6, algorithm='apr'):
    df2 = df.copy()
    df2['high_value'] = df2['price'] > 5000
    trans_by_method = df2.groupby('payment_method')['category'].apply(list).tolist()
    algo = 'apriori' if algorithm.startswith('apr') else 'fpgrowth'
    print("Sadasdas")
    freq, rules = mine_rules(trans_by_method, min_support, min_confidence, algo)
    
    high_methods = df2[df2['high_value']]['payment_method'].value_counts()
    return freq, rules, high_methods

def payment_category_association_v2(df, min_support=0.01, min_confidence=0.6, algorithm='apr'):
    # 对每个订单，收集品类列表，并把支付方式当一项加进去
    orders = (
        df.groupby('order_id')
          .apply(lambda g: g['category'].tolist() + [g['payment_method'].iloc[0]])
          .tolist()
    )
    algo = 'apriori' if algorithm.startswith('apr') else 'fpgrowth'
    freq, rules = mine_rules(orders, min_support, min_confidence, algorithm=algo)
    # 只挑出既含支付方式又含品类的规则
    rules = rules[
        rules['antecedents'].apply(lambda s: any(pm in s for pm in df['payment_method'].unique())) |
        rules['consequents'].apply(lambda s: any(pm in s for pm in df['payment_method'].unique()))
    ]
    # 高频价值支付方式
    high_methods = df[df['price'] > 5000]['payment_method'].value_counts()
    return freq, rules, high_methods


# 4. 时间序列模式
def time_series_patterns(df):
    df2 = df.copy()
    df2['quarter'] = df2['purchase_date'].dt.to_period('Q')
    df2['month']   = df2['purchase_date'].dt.to_period('M')
    df2['weekday'] = df2['purchase_date'].dt.day_name()

    season_counts  = df2.groupby(['quarter','category']).size().unstack(fill_value=0)
    month_counts   = df2.groupby(['month','category']).size().unstack(fill_value=0)
    weekday_counts = df2.groupby(['weekday','category']).size().unstack(fill_value=0)

    df_sorted = df2.sort_values(['order_id','purchase_date'])
    seqs = []
    for _, grp in df_sorted.groupby('order_id')['category']:
        for a, b in zip(grp.tolist()[:-1], grp.tolist()[1:]):
            seqs.append((a, b))
    seq_df = pd.DataFrame(seqs, columns=['A','B'])
    seq_counts = seq_df.value_counts().reset_index(name='count')

    return season_counts, month_counts, weekday_counts, seq_counts

# 5. 退款模式
def refund_pattern_analysis(df, min_support=0.005, min_confidence=0.4, algorithm='apr'):
    refunded = df[df['payment_status'].isin(['已退款','部分退款'])]
    orders = refunded.groupby('order_id')['category'].apply(list).tolist()
    algo = 'apriori' if algorithm.startswith('apr') else 'fpgrowth'
    freq, rules = mine_rules(orders, min_support, min_confidence, algo)
    return freq, rules

# 6. 可视化示例
def plot_seasonality(season_counts):
    season_counts.plot(kind='bar', figsize=(10,7))
    plt.title('季度购买频率')
    plt.xlabel('季度')
    plt.ylabel('购买次数')
    # plt.tight_layout()
    legend = plt.legend()

    legend.set_visible(False)
    plt.savefig("image.png")

# 7. 主流程
def main(parquet_path, products_json_path):
    start_time = time.time()
    df_raw      = load_parquet_data(parquet_path)
    # df_raw=df_raw[:1000]
    products_df = load_product_catalog(products_json_path)
    print(f"原始记录数: {len(df_raw)}")

    df = explode_purchase_history(df_raw, products_df)
    print(f"展开后记录数: {len(df)}")

    # 1. 商品类别关联规则
    print("\n#1 商品类别关联规则")
    fi_cat, rules_cat, rules_elec = category_association(df)
    # 打印
    print(fi_cat)
    print(rules_cat)
    print(rules_elec)
    # 保存到文件
    fi_cat.to_csv('freq_category.csv', index=False)
    rules_cat.to_csv('rules_category.csv', index=False)
    rules_elec.to_csv('rules_elec_category.csv', index=False)
    print("已保存：freq_category.csv, rules_category.csv, rules_elec_category.csv")

    # 2. 支付方式与类别关联
    print("\n#2 支付方式与类别关联")
    fi_pay, rules_pay, high_methods = payment_category_association_v2(df)
    print(fi_pay)
    print(rules_pay)
    print(high_methods)
    # 保存
    fi_pay.to_csv('freq_payment.csv', index=False)
    rules_pay.to_csv('rules_payment.csv', index=False)
    high_methods.to_csv('high_value_payment_methods.csv', header=['count'])
    print("已保存：freq_payment.csv, rules_payment.csv, high_value_payment_methods.csv")

    # 3. 时间序列模式
    print("\n#3 时间序列模式")
    season, month, weekday, seq = time_series_patterns(df)
    plot_seasonality(season)
    print(season)
    print(seq.head())
    # 如果需要保存时间序列统计，也可以：
    season.reset_index().to_csv('season_counts.csv', index=False)
    month.reset_index().to_csv('month_counts.csv', index=False)
    weekday.reset_index().to_csv('weekday_counts.csv', index=False)
    seq.to_csv('sequence_counts.csv', index=False)
    print("已保存：season_counts.csv, month_counts.csv, weekday_counts.csv, sequence_counts.csv")

    # 4. 退款模式
    print("\n#4 退款模式")
    fi_ref, rules_ref = refund_pattern_analysis(df)
    print(fi_ref)
    print(rules_ref)
    # 保存
    fi_ref.to_csv('freq_refund.csv', index=False)
    rules_ref.to_csv('rules_refund.csv', index=False)
    print("已保存：freq_refund.csv, rules_refund.csv")

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='购物历史关联规则挖掘（终极版）')
    parser.add_argument('--input_parquet', default="/home/hfxia/data_trans/30G_data_new/part-00000.parquet")
    parser.add_argument('--products_json', default="/home/hfxia/data_trans/product_catalog.json")
    args = parser.parse_args()
    main(args.input_parquet, args.products_json)
