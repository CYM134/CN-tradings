#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集脚本 - 适配MySQL数据库
生成示例数据到stock_basic和stock_daily_history表
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_simple import DevelopmentConfig
from app.models.stock_basic import StockBasic
from app.models.stock_daily_history import StockDailyHistory

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollector:
    """数据收集器 - MySQL版本"""
    
    def __init__(self):
        """初始化数据库连接"""
        config = DevelopmentConfig()
        self.db_uri = config.SQLALCHEMY_DATABASE_URI
        
        # 创建数据库引擎
        self.engine = create_engine(self.db_uri, pool_pre_ping=True)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"✅ 数据库连接成功: {self.db_uri.split('@')[-1]}")
    
    def generate_sample_stock_info(self):
        """生成示例股票基本信息 - 使用Tushare格式的股票代码"""
        sample_stocks = [
            ('000001.SZ', '000001', '平安银行', '深圳', '银行', '1991-04-03'),
            ('000002.SZ', '000002', '万科A', '深圳', '房地产', '1991-01-29'),
            ('600000.SH', '600000', '浦发银行', '上海', '银行', '1999-11-10'),
            ('600036.SH', '600036', '招商银行', '上海', '银行', '2002-04-09'),
            ('000858.SZ', '000858', '五粮液', '四川', '食品饮料', '1998-04-27'),
            ('002415.SZ', '002415', '海康威视', '浙江', '电子', '2010-05-28'),
            ('300059.SZ', '300059', '东方财富', '上海', '非银金融', '2010-03-19'),
            ('000063.SZ', '000063', '中兴通讯', '广东', '通信', '1997-11-18'),
            ('600519.SH', '600519', '贵州茅台', '贵州', '食品饮料', '2001-08-27'),
            ('601318.SH', '601318', '中国平安', '广东', '非银金融', '2007-03-01')
        ]
        
        return sample_stocks
    
    def collect_stock_info(self):
        """收集股票基本信息到stock_basic表"""
        logger.info("\n" + "="*60)
        logger.info("📥 开始收集股票基本信息...")
        logger.info("="*60)
        
        try:
            sample_stocks = self.generate_sample_stock_info()
            success_count = 0
            
            for ts_code, symbol, name, area, industry, list_date_str in sample_stocks:
                try:
                    # 转换日期格式
                    list_date = datetime.strptime(list_date_str, '%Y-%m-%d').date()
                    
                    # 检查是否已存在
                    existing = self.session.query(StockBasic).filter_by(ts_code=ts_code).first()
                    
                    if existing:
                        # 更新现有记录
                        existing.symbol = symbol
                        existing.name = name
                        existing.area = area
                        existing.industry = industry
                        existing.list_date = list_date
                        logger.info(f"🔄 {ts_code} ({name}): 更新基本信息")
                    else:
                        # 插入新记录
                        stock_basic = StockBasic(
                            ts_code=ts_code,
                            symbol=symbol,
                            name=name,
                            area=area,
                            industry=industry,
                            list_date=list_date
                        )
                        self.session.add(stock_basic)
                        logger.info(f"✅ {ts_code} ({name}): 新增基本信息")
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ {ts_code}: 处理失败 - {str(e)}")
                    continue
            
            # 提交事务
            self.session.commit()
            logger.info(f"\n✅ 股票基本信息收集完成: {success_count}/{len(sample_stocks)}")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"收集股票基本信息失败: {e}")
            return False
    
    def generate_sample_price_data(self, ts_code, days=365):
        """生成示例价格数据（近1年）"""
        # 使用股票代码的哈希作为随机种子，保证每次生成相同
        np.random.seed(hash(ts_code) % 2**32)
        
        # 根据股票不同设置不同的基础价格
        if '600519' in ts_code:  # 贵州茅台
            base_price = 1800.0
        elif '601318' in ts_code:  # 中国平安
            base_price = 55.0
        elif '300059' in ts_code:  # 东方财富
            base_price = 18.0
        elif '000858' in ts_code:  # 五粮液
            base_price = 180.0
        elif '600036' in ts_code or '600000' in ts_code:  # 银行股
            base_price = 35.0
        else:
            base_price = np.random.uniform(10, 100)
        
        # 生成价格序列
        prices = []
        current_price = base_price
        
        for i in range(days):
            # 随机波动（符合A股特点）
            change_pct = np.random.normal(0, 0.02)  # 2%的标准差
            # 限制涨跌幅在±10%以内
            change_pct = max(min(change_pct, 0.10), -0.10)
            current_price *= (1 + change_pct)
            
            # 确保价格为正
            current_price = max(current_price, 0.01)
            
            # 生成OHLC数据
            high = current_price * np.random.uniform(1.0, 1.03)
            low = current_price * np.random.uniform(0.97, 1.0)
            open_price = np.random.uniform(low, high)
            close_price = current_price
            
            # 成交量（手）
            volume = int(np.random.uniform(100000, 10000000))
            # 成交额（千元）
            amount = volume * close_price / 10  # 转换为千元
            
            # 交易日期（往前推）
            trade_date = (datetime.now() - timedelta(days=days-i-1)).date()
            
            # 跳过周末（简化处理）
            if trade_date.weekday() >= 5:
                continue
            
            # 计算涨跌额和涨跌幅
            if len(prices) > 0:
                pre_close = prices[-1]['close']
                change = close_price - pre_close
                pct_chg = (change / pre_close) * 100
            else:
                pre_close = current_price
                change = 0
                pct_chg = 0
            
            prices.append({
                'ts_code': ts_code,
                'trade_date': trade_date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'pre_close': round(pre_close, 2),
                'change': round(change, 2),
                'pct_chg': round(pct_chg, 2),
                'vol': volume,
                'amount': round(amount, 2)
            })
        
        return prices
    
    def collect_price_data(self, ts_codes=None, days=365):
        """收集股票价格数据到stock_daily_history表（近1年）"""
        if ts_codes is None:
            ts_codes = [info[0] for info in self.generate_sample_stock_info()]
        
        logger.info("\n" + "="*60)
        logger.info(f"📥 开始收集{len(ts_codes)}只股票的日线数据...")
        logger.info(f"📅 时间范围: 近{days}天（约1年）")
        logger.info("="*60)
        
        try:
            total_records = 0
            
            for ts_code in ts_codes:
                try:
                    logger.info(f"📊 收集 {ts_code} 的价格数据...")
                    
                    price_data = self.generate_sample_price_data(ts_code, days)
                    count = 0
                    
                    for data in price_data:
                        # 检查是否已存在
                        existing = self.session.query(StockDailyHistory).filter_by(
                            ts_code=data['ts_code'],
                            trade_date=data['trade_date']
                        ).first()
                        
                        if existing:
                            # 更新现有记录
                            existing.open = data['open']
                            existing.high = data['high']
                            existing.low = data['low']
                            existing.close = data['close']
                            existing.pre_close = data['pre_close']
                            existing.change = data['change']
                            existing.pct_chg = data['pct_chg']
                            existing.vol = data['vol']
                            existing.amount = data['amount']
                        else:
                            # 插入新记录
                            daily_data = StockDailyHistory(
                                ts_code=data['ts_code'],
                                trade_date=data['trade_date'],
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                pre_close=data['pre_close'],
                                change=data['change'],
                                pct_chg=data['pct_chg'],
                                vol=data['vol'],
                                amount=data['amount']
                            )
                            self.session.add(daily_data)
                        
                        count += 1
                    
                    # 每个股票提交一次
                    self.session.commit()
                    total_records += count
                    logger.info(f"✅ {ts_code}: 已保存 {count} 条数据")
                    
                except Exception as e:
                    self.session.rollback()
                    logger.error(f"❌ {ts_code}: 收集失败 - {str(e)}")
                    continue
            
            logger.info(f"\n✅ 日线数据收集完成，共 {total_records} 条记录")
            return True
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"收集价格数据失败: {e}")
            return False
    
    def verify_data(self, ts_codes=None):
        """验证下载的数据"""
        if ts_codes is None:
            ts_codes = [info[0] for info in self.generate_sample_stock_info()]
        
        logger.info("\n" + "="*60)
        logger.info("🔍 验证下载的数据...")
        logger.info("="*60)
        
        for ts_code in ts_codes:
            # 查询基本信息
            basic = self.session.query(StockBasic).filter_by(ts_code=ts_code).first()
            if not basic:
                logger.warning(f"⚠️  {ts_code}: 未找到基本信息")
                continue
            
            # 查询历史数据条数
            count = self.session.query(StockDailyHistory).filter_by(ts_code=ts_code).count()
            
            # 查询最新和最早日期
            from sqlalchemy import desc, asc
            latest = self.session.query(StockDailyHistory).filter_by(ts_code=ts_code).order_by(
                desc(StockDailyHistory.trade_date)
            ).first()
            
            earliest = self.session.query(StockDailyHistory).filter_by(ts_code=ts_code).order_by(
                asc(StockDailyHistory.trade_date)
            ).first()
            
            if latest and earliest:
                logger.info(
                    f"✅ {ts_code} ({basic.name}): "
                    f"{count} 条数据 | "
                    f"范围: {earliest.trade_date} ~ {latest.trade_date} | "
                    f"最新价: ¥{latest.close} | "
                    f"涨跌幅: {latest.pct_chg:+.2f}%"
                )
            else:
                logger.warning(f"⚠️  {ts_code} ({basic.name}): 无历史数据")
    
    def close(self):
        """关闭数据库连接"""
        self.session.close()
        logger.info("\n✅ 数据库连接已关闭")
    
    def run_data_collection(self):
        """执行数据收集"""
        logger.info("\n" + "="*70)
        logger.info("🚀 开始执行数据收集任务...")
        logger.info("="*70)
        
        try:
            # 1. 收集股票基本信息
            if not self.collect_stock_info():
                logger.error("❌ 股票基本信息收集失败")
                return False
            
            # 2. 收集近1年的价格数据
            if not self.collect_price_data(days=365):
                logger.error("❌ 价格数据收集失败")
                return False
            
            # 3. 验证数据
            self.verify_data()
            
            logger.info("\n" + "="*70)
            logger.info("✅ 数据收集任务执行完成")
            logger.info("💡 提示: 现在可以访问前端「股票列表」页面查看数据")
            logger.info("="*70)
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据收集任务执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    
    # 执行数据收集
    collector = DataCollector()
    success = collector.run_data_collection()
    
    if success:
        print("✅ 数据收集执行成功")
        sys.exit(0)
    else:
        print("❌ 数据收集执行失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
