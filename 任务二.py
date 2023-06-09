import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')  # 不显示警告

def process_dates(data):
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    columns = data.columns.tolist()
    if 'Date' in columns:
        data['date'] = pd.to_datetime(data['Date'], format="%Y%m%d")
    return data

# def time_transport(data): #领券日期处理
#     # 时间处理转换
#     data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
#     return data['date_received']

def discount_rate(x):
    if ':' in str(x):
        return 1 - float(str(x).split(':')[1]) / (float(str(x).split(':')[0]))
    else:
        return float(x)

def flag_of_manjian(x):
    if ':' in str(x):
        return 1
    else:
        return 0

def manjian_at_least_cost(x):
    if ':' in str(x):
        return int(str(x).split(':')[0])
    else:
        return -1

def offline_prepare(data1):
    data = data1.copy()
    #填充操作
    data['Distance'].fillna(-1, inplace=True);data['Coupon_id'].fillna(-1, inplace=True)
    #折扣率操作
    data['Discount_rate'].fillna(-1, inplace=True);data['discount_rate'] = data['Discount_rate'].map(discount_rate)

    data['flag_of_manjian'] = data['Discount_rate'].map(flag_of_manjian)
    data['manjian_at_least_cost'] = data['Discount_rate'].map(manjian_at_least_cost)
    #日期操作
    data = process_dates(data)
    return data



def feat_prepare(history_field, label_field, keys, field):
    data = history_field.copy()
    # data[['Date', 'date', 'date_received', 'Date_received']] = data[['Date', 'date', 'date_received', 'Date_received']].fillna(-1,inplace=True)
    data['Date'].fillna(-1, inplace=True);data['date'].fillna(-1, inplace=True);data['date_received'].fillna(-1, inplace=True);data['Date_received'].fillna(-1, inplace=True);data['cnt'] = 1
    data['Coupon_id'] = data['Coupon_id'].map(int);data['Date_received'] = data['Date_received'].map(int)
    feat = label_field[keys].drop_duplicates(keep='first')
    return data, feat

def mer(data1, data2, key, fill_value):
    data1 = pd.merge(data1, data2, on = key, how = 'left')
    data1.fillna(fill_value, downcast='infer', inplace = True)
    return data1

def get_receive_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_cnt']
    return date

def get_receive_max_distance(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Distance'].max()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_max_distance']
    return date

def get_receive_min_distance(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Distance'].min()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_min_distance']
    return date

def get_receive_mean_distance(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Distance'].mean()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_mean_distance']
    return date
def get_receive_and_consume_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_cnt']
    return date
def get_receive_not_consume_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (
        data['Date'].map(lambda x: x == -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_not_consume_cnt']
    return date
def calculate_rate(x, y):
    if y == 0:
        return 0
    else:
        return x / y
def get_receive_and_consume_differ_Merchant_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Merchant_id'].apply(lambda x : len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_differ_Merchant_cnt']
    return date
def get_receive_differ_Merchant_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Merchant_id'].apply(lambda x: len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_differ_Merchant_cnt']
    return date
def get_receive_differ_coupon_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_differ_coupon_cnt']
    return date
def calculate_gap(x, y):
    if x != -1 and y != -1:
        return (x - y).days
    else:
        return 0

def is_consume_15day(x):
    if x >= 0 and x <= 15:
        return 1
    else:
        return 0

def get_receive_and_consume_in15day_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_in15day_cnt']
    return date

def get_receive_not_consume_in15day_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) &
              (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x: x == 0))]
    date = pd.DataFrame(fd.groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_not_consume_in15day_cnt']
    return date

def get_receive_and_consume_in15day_differ_Coupon_cnt(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) &
              (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) &
              (data[prefixs + 'is_consume_15day'].map(lambda x: x == 1)) & (data['discount_rate'].map(lambda x: x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt']
    return date


def user_feature(history_field, label_field):       #1111111111111
    #主键和特征预处理
    keys = ['User_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, u_feat = feat_prepare(history_field, label_field, keys, field)
    history_feat = label_field.copy()
    # 1.用户领卷数   1111
    tmp = get_receive_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 2.用户领取优惠卷的最大距离   111
    tmp = get_receive_max_distance(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #3.用户领取优惠卷的最小距离   111
    tmp = get_receive_min_distance(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #4.用户领取优惠卷的平均距离  111
    tmp = get_receive_mean_distance(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 5.用户领卷并消费数      1111
    tmp = get_receive_and_consume_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 6.用户领卷未消费数  111
    tmp = get_receive_not_consume_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 7.用户领卷核销率   1111
    u_feat[prefixs + 'receive_and_consume_rate'] = list(map(calculate_rate, u_feat[prefixs + 'receive_and_consume_cnt'], u_feat[prefixs + 'receive_cnt']))

    # 用户领券的未核销率
    u_feat[prefixs + 'receive_not_consume_rate'] = list(map(calculate_rate, u_feat[prefixs + 'receive_not_consume_cnt'], u_feat[prefixs + 'receive_cnt']))

    # 8.在多少不同商家领取并消费优惠卷
    tmp = get_receive_and_consume_differ_Merchant_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 9.在多少不同商家领取优惠卷    1111
    tmp = get_receive_differ_Merchant_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #10.在多少不同商家领取优惠卷核销率
    u_feat[prefixs + 'receive_differ_Merchant_consume_rate'] = list(map(calculate_rate,
                                                            u_feat[prefixs + 'receive_and_consume_differ_Merchant_cnt'],u_feat[prefixs + 'receive_differ_Merchant_cnt']))

    #11.用户领取不同优惠卷数量
    tmp = get_receive_differ_coupon_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #13.用户领卷日期与消费日期之间的时间差
    data[prefixs + 'gap'] = list(map(calculate_gap, data['date'], data['date_received']))
    data[prefixs + 'is_consume_15day'] = list(map(is_consume_15day, data[prefixs + 'gap']))

    #14.用户领取优惠卷对特定商家15天内核销的次数    111
    tmp = get_receive_and_consume_in15day_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)


    #15.用户领取优惠卷但15天内没有核销  1111
    tmp = get_receive_not_consume_in15day_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #16.用户核销数与用户对领卷商家15天内的核销数的比重  1111
    u_feat[prefixs + 'receive_and_consume_with_Merhcant_consume_in15days_rate'] = list(map(calculate_rate,
                                                                            u_feat[prefixs + 'receive_and_consume_cnt'], u_feat[prefixs + 'receive_and_consume_in15day_cnt']))

    #17.用户15天内核销的不同优惠卷的数量     1111
    tmp = get_receive_and_consume_in15day_differ_Coupon_cnt(data, keys, prefixs)
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #18.用户对不同卷在15天内的核销率    1111
    u_feat[prefixs + 'differ_Coupon_in15day_consume_rate'] = list(map(calculate_rate,
                                                        u_feat[prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt'], u_feat[prefixs + 'receive_differ_coupon_cnt']))

    #19.用户15天内核销的最大折扣率  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 1)) & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys[0])['discount_rate'].max()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_max_discount_rate']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #20.用户15天内核销的最小折扣率  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x: x == 1)) & (data['discount_rate'].map(lambda x: x != -1))].groupby(keys[0])['discount_rate'].min()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_min_discount_rate']
    u_feat = mer(u_feat, tmp, keys[0], 0)


    history_feat = pd.merge(history_feat, u_feat, on=keys, how='left')
    return history_feat



def get_receive_cnt_m(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_cnt']
    return date

def get_common_consume_cnt_m(data, keys, prefixs):
    fd = data[data['Date'].map(lambda x : x != -1)]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'common_consume_cnt']
    return date
def get_receive_differ_User_cnt_m(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['User_id'].apply(lambda x : len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_differ_User_cnt']
    return date

def get_receive_and_consume_cnt_m(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) &
              (data['Date_received'].map(lambda x : x != -1)) &
              (data['Date'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_cnt']
    return date

def get_differ_Coupon_cnt_m(data, keys, prefixs):
    fd = data[data['Coupon_id'].map(lambda x : x != -1)]
    date = pd.DataFrame(fd.groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'differ_Coupon_cnt']
    return date

def get_receive_and_consume_differ_User_cnt_m(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) &
              (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['User_id'].apply(lambda x : len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_differ_User_cnt']
    return date

def get_receive_and_consume_in15day_min_gap_m(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) &
              (data['Date'].map(lambda x: x != -1)) &
              (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))]
    date = pd.DataFrame(fd.groupby(keys[0])[prefixs + 'gap'].min()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_in15day_min_gap']
    return date
def get_receive_and_consume_in15day_mean_gap_m(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) &
              (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) &
              (data[prefixs + 'consume_in15day'].map(lambda x : x == 1))]
    date = pd.DataFrame(fd.groupby(keys[0])[prefixs + 'gap'].mean()).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_in15day_mean_gap']
    return date
def get_receive_and_consume_in15day_differ_Coupon_cnt_m(data, keys, prefixs):
    fd =data[(data['Coupon_id'].map(lambda x: x != -1)) &
             (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) &
             (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))]
    date = pd.DataFrame(fd.groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    date.columns = [keys[0], prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt']
    return date

#商家特征群
def merchant_feature(history_field, label_field):       #11  11111111111
    # 主键和特征预处理
    keys = ['Merchant_id']
    field = 'history_field'
    prefixs = 'history_field_' +'_'.join(keys) + '_'
    data, m_feat = feat_prepare(history_field, label_field, keys, field)
    history_feat = label_field.copy()
    #1.商家的优惠券被领取的次数   111
    tmp = get_receive_cnt_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #2.商家总共被消费次数
    tmp = get_common_consume_cnt_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #3.商家被不同客户领取的次数  111
    tmp = get_receive_differ_User_cnt_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #4.商家的券被核销的次数  111
    tmp = get_receive_and_consume_cnt_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #5.商家的券被核销率     111
    m_feat[prefixs + 'received_and_consumed_rate'] = list(map(calculate_rate ,
                                                              m_feat[prefixs + 'receive_and_consume_cnt'],m_feat[prefixs + 'receive_cnt']))

    #6.商家提供的不同优惠券数  111
    tmp = get_differ_Coupon_cnt_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)


    #7.商家被不同客户核销的次数
    tmp = get_receive_and_consume_differ_User_cnt_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)


    #8.商家优惠卷平均每张被核销多少次  111
    m_feat[prefixs + 'Coupon_consume_mean_cnt'] = list(map(calculate_rate,
                                                           m_feat[prefixs + 'receive_and_consume_cnt'], m_feat[prefixs + "differ_Coupon_cnt"]))

    # 9.商家优惠卷被领取后15天内被核销的次数
    data[prefixs + 'gap'] = list(map(calculate_gap, data['date'], data['date_received']))
    data[prefixs + 'consume_in15day'] = list(map(is_consume_15day , data[prefixs + 'gap']))

    #29.商家优惠卷核销平均间隔天数 1111
    tmp = get_receive_and_consume_in15day_mean_gap_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #30.商家优惠卷核销最小间隔天数  111
    tmp = get_receive_and_consume_in15day_min_gap_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #31.商家15天内核销的不同种类优惠卷数量
    tmp = get_receive_and_consume_in15day_differ_Coupon_cnt_m(data, keys, prefixs)
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #32.商家15天内核销不同优惠卷占所有优惠卷比重 1111
    m_feat[prefixs + 'consume_differ_coupon_in15day_with_all_coupon_rate'] = list(map(calculate_rate,
                                                                            m_feat[prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt'], m_feat[prefixs + 'differ_Coupon_cnt']))


    history_feat = pd.merge(history_feat, m_feat, on = keys, how = 'left')
    return history_feat

#优惠卷特征群
def get_received_cnt_c(data, keys, prefixs):
    fd = data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'received_cnt']
    return date
def get_received_and_consumed_cnt_15_c(data, keys, prefixs):
    fd = data[data['label'].map(lambda x : x == 1)]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'received_and_consumed_cnt_15']
    return date
def get_received_not_consumed_cnt_15_c(data, keys, prefixs):
    fd = data[data['label'].map(lambda x : x != 1)]
    date = pd.DataFrame(fd.groupby(keys[0])['cnt'].count()).reset_index()
    date.columns = [keys[0], prefixs + 'received_not_consumed_cnt_15']
    return date

def get_consumed_mean_time_gap_15_c(data, keys, prefixs):
    fd = data[(data[prefixs + 'is_consume_15day'] == 1)]
    date = pd.DataFrame(fd.groupby(keys[0])[prefixs + 'gap'].mean()).reset_index()
    date.columns = [keys[0], prefixs + 'consumed_mean_time_gap_15']
    return date

def coupon_feature(history_field, label_field):
    # 主键和特征预处理
    keys = ['Coupon_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, c_feat = feat_prepare(history_field, label_field, keys, field)
    history_feat = label_field.copy()
    #1.优惠卷被领取的次数  111
    tmp = get_received_cnt_c(data, keys, prefixs)
    c_feat = mer(c_feat, tmp, keys[0], 0)

    #2.优惠卷15天内被核销次数  111
    tmp = get_received_and_consumed_cnt_15_c(data, keys, prefixs)
    c_feat = mer(c_feat, tmp, keys[0], 0)

    #3.优惠卷15天内未核销次数 111
    tmp = get_received_not_consumed_cnt_15_c(data, keys, prefixs)
    c_feat = mer(c_feat, tmp, keys[0], 0)

    #4.15天内核销/未核销  111
    c_feat[prefixs + 'consume_with_not_consume_rate_in15day'] = list(map(calculate_rate ,
                                        c_feat[prefixs + 'received_and_consumed_cnt_15'], c_feat[prefixs + 'received_not_consumed_cnt_15']))

    #3.优惠卷15天内被核销率  111
    c_feat[prefixs + 'received_and_consumed_rate_15'] = list(map(calculate_rate,
                                        c_feat[prefixs + 'received_and_consumed_cnt_15'], c_feat[prefixs + 'received_cnt']))

    #5.优惠卷15天内未核销率  111
    c_feat[prefixs + 'received_not_consumed_rate_15'] = list(map(calculate_rate,
                                        c_feat[prefixs + 'received_not_consumed_cnt_15'], c_feat[prefixs + 'received_cnt']))

    #4.优惠卷15天内被核销的平均时间间隔  111
    data[prefixs + 'gap'] = list(map(calculate_gap, data['date'], data['date_received']))
    data[prefixs + 'is_consume_15day'] = list(map(is_consume_15day , data[prefixs + 'gap']))
    tmp1 = get_consumed_mean_time_gap_15_c(data, keys, prefixs)
    c_feat = mer(c_feat, tmp1, keys[0], 0)



    history_feat = pd.merge(history_feat, c_feat, on=keys, how='left')
    return history_feat

#用户商家交叉特征群
def get_received_cnt_um(data, keys, prefixs, uc_feat):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) &
             (data['Date_received'].map(lambda x: x != -1))]
    tmp = pd.DataFrame(fd.groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'received_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)
    return uc_feat
def get_receive_and_consume_cnt_um(data, keys, prefixs, uc_feat):
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (
        data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1))]
    tmp = pd.DataFrame(fd.groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)
    return uc_feat
def get_receive_and_not_consume_cnt_um(data, keys, prefixs, uc_feat) :
    fd = data[(data['Coupon_id'].map(lambda x: x != -1)) & (
        data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x == -1))]
    tmp = pd.DataFrame(fd.groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_not_consume_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)
    return uc_feat



def user_merchant_feature(history_field, label_field):
    # 主键和特征预处理
    keys = ['User_id', 'Merchant_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, uc_feat = feat_prepare(history_field, label_field, keys, field)
    history_feat = label_field.copy()
    # 1.用户领取该商家优惠卷数目    111
    uc_feat = get_received_cnt_um(data, keys, prefixs, uc_feat)

    # 2.用户核销该商家的优惠卷数目     111
    uc_feat = get_receive_and_consume_cnt_um(data, keys, prefixs, uc_feat)

    # 3.用户领取该商家优惠卷后不核销次数    111
    uc_feat = get_receive_and_not_consume_cnt_um(data, keys, prefixs, uc_feat)

    # 4.用户领取商家核销率      111
    uc_feat[prefixs + 'receive_and_consume_rate'] = list(map(calculate_rate,
                        uc_feat[prefixs + 'receive_and_consume_cnt'], uc_feat[prefixs + 'received_cnt']))

    # 5.用户未核销率         1111
    uc_feat[prefixs + 'receive_not_consume_rate'] = list(map(calculate_rate,
                        uc_feat[prefixs + 'receive_and_not_consume_cnt'],uc_feat[prefixs + 'received_cnt']))

    history_feat = pd.merge(history_feat, uc_feat, on=keys, how='left')
    return history_feat


#其他特征提取
def get_User_receive_cnt(data, prefixs, feature):
    data1 = data.groupby(['User_id', 'Coupon_id'])['cnt'].count()
    tmp = pd.DataFrame(data1)
    tmp = tmp.reset_index()
    tmp.columns = ['User_id', 'Coupon_id', prefixs + 'User_Coupon_receive_cnt']
    feature1 = mer(feature, tmp, ['User_id', 'Coupon_id'], 0)
    return feature1

def get_User_Coupon_receive_cnt(data, prefixs, feature): #用户领取特定优惠卷的数目
    data2 = data.groupby(['User_id', 'Coupon_id'])['cnt'].count()
    tmp = pd.DataFrame(data2)
    tmp = tmp.reset_index()
    tmp.columns = ['User_id', 'Coupon_id', prefixs + 'User_Coupon_receive_cnt']
    feature2 = mer(feature, tmp, ['User_id', 'Coupon_id'], 0)
    return feature2
def get_User_receive_merchant_cnt(data, prefixs, feature): #用户领取特定商家的优惠卷数目
    data3 = data.groupby(['User_id', 'Merchant_id'])['cnt'].count()
    tmp = pd.DataFrame(data3)
    tmp = tmp.reset_index()
    tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_receive_merchant_cnt']
    feature3 = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    return feature3
def get_Merchant_receive_cnt(data, prefixs, feature): #商家被领取的优惠卷数目
    data4 = data.groupby('Merchant_id')['cnt'].count()
    tmp = pd.DataFrame(data4)
    tmp =tmp.reset_index()
    tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_cnt']
    feature4 = mer(feature, tmp, 'Merchant_id', 0)
    return feature4
def user_merchant_is_first(data, prefixs, feature): #用户是否第一次领取该商家的优惠卷
    tmp = data.copy()
    tmp = tmp.sort_values(by=['User_id', 'Merchant_id', 'Date_received'])
    tmp1 = tmp.drop_duplicates(subset=['User_id', 'Merchant_id'], keep='first'); tmp1 = tmp1[['User_id', 'Merchant_id', 'Date_received']]
    tmp1[prefixs + 'User_id_first_received_merchant_flag'] = 1
    result = mer(feature, tmp1, ['User_id', 'Merchant_id', 'Date_received'], 0)
    return result

def user_merchant_is_last(data, prefixs, feature): #用户是否最后一次领取该商家优惠卷
    tmp = data.copy()
    tmp1 = tmp.sort_values(by=['User_id', 'Merchant_id', 'Date_received']).reset_index().drop(columns='index', axis=1)
    tmp1 = tmp1.drop_duplicates(subset=['User_id', 'Merchant_id'], keep='last'); tmp1 = tmp1[['User_id', 'Merchant_id', 'Date_received']]
    tmp1[prefixs + 'User_id_last_received_merchant_flag'] = 1
    result = mer(feature, tmp1, ['User_id', 'Merchant_id', 'Date_received'], 0)
    return result
def user_coupon_is_first(data, prefixs, feature):
    tmp = data.copy()
    tmp1 = tmp.sort_values(by=['User_id', 'Date_received']).reset_index().drop(columns='index', axis=1)
    tmp1 = tmp1.drop_duplicates(['User_id'], keep='first'); tmp1 = tmp1[['User_id', 'Date_received']]
    tmp1[prefixs + 'User_id_first_receive_flag'] = 1
    result = mer(feature, tmp1, ['User_id', 'Date_received'], 0)
    return result
def user_coupon_is_last(data, prefixs, feature):
    tmp = data.copy()
    tmp1 = tmp.sort_values(by=['User_id', 'Date_received']).reset_index().drop(columns='index', axis=1)
    tmp1 = tmp1.drop_duplicates(['User_id'], keep='last'); tmp1 = tmp1[['User_id', 'Date_received']]
    tmp1[prefixs + 'User_id_last_receive_flag'] = 1
    result = mer(feature, tmp1, ['User_id', 'Date_received'], 0)
    return result


def other_feature(label_field):      #111111

    prefixs = 'other_field_'
    feature = label_field.copy()

    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].astype(int); data['Date_received'] = data['Date_received'].astype(int)
    data.fillna({'Date': -1, 'date_received': -1, 'Date_received': -1}, inplace=True) ; data['cnt'] = 1
    data.drop_duplicates(inplace=True)
    data = data[data['Coupon_id'].map(lambda x: x != -1)]

    #1.用户领取所有优惠卷的数目
    feature = get_User_receive_cnt(data, prefixs, feature)
    #2.用户领取特定优惠卷的数目
    feature = get_User_Coupon_receive_cnt(data, prefixs, feature)
    #3.用户领取特定商家的优惠卷数目
    feature = get_User_receive_merchant_cnt(data, prefixs, feature)
    #4.商家被领取的优惠卷数目
    feature = get_Merchant_receive_cnt(data, prefixs, feature)
    #5.用户是否第一次领取该商家的优惠卷
    feature = user_merchant_is_first(data, prefixs, feature)
    #6.用户是否最后一次领取该商家优惠卷
    feature = user_merchant_is_last(data, prefixs, feature)
    #7.用户是否第一次领取优惠卷
    feature = user_coupon_is_first(data, prefixs, feature)
    #8.用户是否最后一次领取优惠卷
    feature = user_coupon_is_last(data, prefixs, feature)
    return feature



def common_characters(*dataframes):
    common_characters = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_characters &= set(df.columns)
    return list(common_characters)

def merge_dataframes(label_field, *dataframes):
    label_field.index = range(len(label_field)); common_chars = common_characters(*dataframes)
    for df in dataframes:
        df.drop(common_chars, axis=1, inplace=True)
    dataset = pd.concat([label_field] + list(dataframes), axis=1)
    return dataset



def preprocess_dataset(dataset):
    if 'Date' in dataset.columns.tolist():
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist();dataset.drop(['label'], axis=1, inplace=True);dataset['label'] = label
    else:
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)

    dataset['User_id'] = dataset['User_id'].map(int);dataset['Coupon_id'] = dataset['Coupon_id'].map(int);dataset['Date_received'] = dataset['Date_received'].map(int);dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)

    dataset.drop_duplicates(keep='first', inplace=True);dataset.index = range(len(dataset))
    return dataset

def dataprocess(history_field, online_field, label_field):  #进行数据的处理
    history_feat = user_feature(history_field, label_field);merchant_feat = merchant_feature(history_field, label_field)
    coupon_feat = coupon_feature(history_field, label_field);um_feat = user_merchant_feature(history_field, label_field)
    other_feat = other_feature(label_field)


    dataset = merge_dataframes(label_field, history_feat, other_feat, merchant_feat, coupon_feat, um_feat)

    dataset = preprocess_dataset(dataset)

    return dataset



def get_params():  #参考baseline里面的参数
    return {'booster': 'gbtree','objective': 'binary:logistic','eval_metric': 'auc',
            'silent': 1,
            'eta': 0.01,
            'max_depth': 5,
            'min_child_weight': 1,'gamma': 0,'lambda': 1,
            'colsample_bylevel': 0.7,'colsample_bytree': 0.7,'subsample': 0.9,'scale_pos_weight': 1,'tree_method': 'gpu_hist','predictor': 'gpu_predictor'}

def model_xgb(train, test):
    params = get_params()
    tr = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    te = test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1)
    train_data = xgb.DMatrix(tr, label=train['label']);test_data = xgb.DMatrix(te);watch = [(train_data, 'train')]

    model = xgb.train(params, train_data, num_boost_round=1000, evals=watch, early_stopping_rounds=50)
    predict = pd.DataFrame(model.predict(test_data, validate_features=False), columns=['prob'])

    return predict


#数据打标
def get_label(dataset): #对数据进行打标
    #索取源数据为后续准备
    data = dataset.copy()
    # 打标:领券后15天内消费为1,否则为0
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],data['date_received']))
    # 返回
    return data


# 划分数据集
# def split_data(data):
#     print("************划分训练集************")
#     train_history_data = data[data['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
#     train_middle_data = data[data['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
#     train_label_data = data[data['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
#     print("************划分验证集************")
#     validate_history_data = data[
#         data['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
#     validate_middle_data = data[data['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
#     validate_label_data = data[
#         data['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
#     print("************划分测试集************")
#     test_history_data = data[data['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
#     test_middle_data = data[data['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
#     test_label_data = test_data.copy()  # [20160701,20160801)
#
#     return train_history_data, train_middle_data, train_label_data, validate_history_data, validate_middle_data, validate_label_data, test_history_data, test_middle_data, test_label_data
def split_data(data):  #划分数据集
    print("************划分训练集************")
    train_history_data = data[(data['date_received']>='2016/3/2') & (data['date_received']<'2016/5/1')]  # [20160302,20160501)
    train_middle_data = data[(data['date']>='2016/5/1')& (data['date']<'2016/5/16')]  # [20160501,20160516)
    train_label_data = data[(data['date_received']>='2016/5/16') & (data['date_received']<'2016/6/16')]  # [20160516,20160616)
    print("************划分验证集************")
    validate_history_data = data[(data['date_received']>='2016/1/16') & (data['date_received']<'2016/3/16')]  # [20160116,20160316)
    validate_middle_data = data[(data['date']>='2016/3/16')& (data['date']<'2016/3/31')]  # [20160316,20160331)
    validate_label_data = data[(data['date_received']>='2016/3/31') & (data['date_received']<'2016/5/1')]  # [20160331,20160501)
    print("************划分测试集************")
    test_history_data = data[(data['date_received']>='2016/4/17') & (data['date_received']<'2016/6/16')]  # [20160417,20160616)
    test_middle_data = data[(data['date']>='2016/6/16')& (data['date']<'2016/7/1')]  # [20160616,20160701)
    test_label_data = test_data.copy()  # [20160701,20160801)

    return train_history_data, train_middle_data, train_label_data, validate_history_data, validate_middle_data, validate_label_data, test_history_data, test_middle_data, test_label_data

if __name__ == '__main__':
    #加载数据
    train_data = pd.read_csv("D:\桌面\任务二\ccf_offline_stage1_train.csv")
    test_data = pd.read_csv("D:\桌面\任务二\ccf_offline_stage1_test_revised.csv")
    online_data = pd.read_csv("D:\桌面\任务二\ccf_online_stage1_train.csv")

    # 数据预处理
    print("************数据正在进行预处理************")
    train_data = offline_prepare(train_data)
    test_data = offline_prepare(test_data)
    #为数据进行打标
    print("************数据进行打标中************")
    train_data = get_label(train_data)
    # 划分区间
    train_history_data, train_middle_data, train_label_data, validate_history_data, validate_middle_data, validate_label_data, test_history_data, test_middle_data, test_label_data = split_data(
        train_data)

    # 构造数据集
    print("************构造训练集中************")
    train1 = dataprocess(train_history_data, online_data, train_label_data)
    print("************构造验证集中************")
    train2 = dataprocess(validate_history_data, online_data, validate_label_data)
    print("************构造测试集中************")
    test = dataprocess(test_history_data, online_data, test_label_data)

    # 连接数据集
    train = pd.concat([train1, train2], axis=0)

    result = model_xgb(train ,test)
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], result], axis=1)

    result.to_csv('output_result.csv', index=False, header=None)