import pandas as pd

def generate_purchase_seq():
  dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
  user_balance = pd.read_csv('./data/user_balance_table.csv', parse_dates=['report_date'],
                               index_col='report_date', date_parser=dateparse)
  
  df = user_balance.groupby(['report_date'])['total_purchase_amt'].sum()
  purchase_seq = pd.Series(df, name='value')

  purchase_seq_train = purchase_seq['2014-04-01':'2014-07-31']
  purchase_seq_test = purchase_seq['2014-08-01':'2014-08-10']

  purchase_seq_train.to_csv(path_or_buf='./data/purchase_seq_train.csv', header=True)
  purchase_seq_test.to_csv(path_or_buf='./data/purchase_seq_test.csv', header=True)
  
generate_purchase_seq()