import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_simple import DevelopmentConfig
from sqlalchemy import create_engine, text

config = DevelopmentConfig()
engine = create_engine(config.SQLALCHEMY_DATABASE_URI)

with engine.connect() as conn:
    result1 = conn.execute(text('SELECT COUNT(*) as cnt FROM stock_basic'))
    count1 = result1.scalar()
    print(f'stock_basic 记录数: {count1}')
    
    result2 = conn.execute(text('SELECT COUNT(*) as cnt FROM stock_daily_history'))
    count2 = result2.scalar()
    print(f'stock_daily_history 记录数: {count2}')
    
    # 显示股票列表
    result3 = conn.execute(text('SELECT ts_code, name FROM stock_basic LIMIT 10'))
    print('\n股票列表:')
    for row in result3:
        print(f'  {row.ts_code} - {row.name}')
    
    # 显示最近的数据
    if count2 > 0:
        result4 = conn.execute(text('''
            SELECT ts_code, trade_date, close, pct_chg 
            FROM stock_daily_history 
            ORDER BY trade_date DESC 
            LIMIT 5
        '''))
        print('\n最近的交易数据:')
        for row in result4:
            print(f'  {row.ts_code} | {row.trade_date} | 收盘: {row.close} | 涨跌幅: {row.pct_chg}%')
