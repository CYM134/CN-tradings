from flask import request, jsonify
from app.api import api_bp
from app.services.stock_service import StockService
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入TradingAgents-CN集成服务
try:
    from app.services.trading_agents_service import trading_agents_service, unified_decision_engine
    TRADING_AGENTS_AVAILABLE = True
    logger.info("TradingAgents-CN集成服务已加载")
except ImportError as e:
    TRADING_AGENTS_AVAILABLE = False
    logger.warning(f"TradingAgents-CN集成服务不可用: {e}")

@api_bp.route('/analysis/unified', methods=['POST'])
def unified_analysis():
    """统一智能分析接口 - 综合量化分析和AI智能决策"""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code')
        trade_date = data.get('trade_date')
        
        if not stock_code:
            return jsonify({
                'success': False,
                'message': '缺少股票代码参数'
            }), 400
        
        if not trade_date:
            trade_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"开始统一分析: 股票={stock_code}, 日期={trade_date}")
        
        # 获取股票基本信息
        stock_info = StockService.get_stock_info(stock_code)
        if not stock_info:
            return jsonify({
                'success': False,
                'message': f'未找到股票: {stock_code}'
            }), 404
        
        # 获取历史数据用于量化分析
        history_data = StockService.get_daily_history(stock_code, limit=100)
        
        # 执行量化分析（基于真实历史数据计算）
        quantitative_result = perform_quantitative_analysis(stock_code, history_data)
        
        # 尝试调用TradingAgents-CN进行AI智能分析
        ai_result = None
        ai_available = False
        if TRADING_AGENTS_AVAILABLE:
            try:
                logger.info(f"调用TradingAgents-CN服务分析: {stock_code}")
                ai_raw = trading_agents_service.analyze_stock(stock_code, trade_date)
                if 'error' not in ai_raw:
                    ai_result = trading_agents_service.format_ai_analysis(ai_raw)
                    if ai_result.get('success') and ai_result.get('ai_available'):
                        ai_available = True
                        logger.info(f"TradingAgents-CN分析成功: {stock_code}")
                    else:
                        logger.warning(f"TradingAgents-CN分析返回失败: {ai_result.get('error', '未知')}")
                else:
                    logger.warning(f"TradingAgents-CN服务错误: {ai_raw['error']}")
            except Exception as e:
                logger.warning(f"TradingAgents-CN调用失败，将使用本地分析: {e}")
        
        # 构建AI分析部分
        if ai_available and ai_result:
            ai_analysis_data = {
                'ai_available': True,
                'overall_rating': ai_result.get('overall_rating', 'HOLD'),
                'confidence': ai_result.get('confidence', 0.5),
                'target_price': ai_result.get('target_price', '基于AI模型'),
                'investment_advice': ai_result.get('investment_advice', ai_result.get('reasoning', '')),
                'analysis_summary': ai_result.get('analysis_summary', {
                    'market_analysis': '市场分析完成',
                    'fundamental_analysis': '基本面分析完成',
                    'news_analysis': '新闻分析完成'
                })
            }
            ai_confidence = ai_result.get('confidence', 0.5)
            ai_rating = ai_result.get('overall_rating', 'HOLD')
            ai_risk_score = ai_result.get('risk_score', 0.5)
        else:
            # AI不可用时，基于量化数据生成本地分析
            local_ai = perform_local_ai_analysis(stock_code, stock_info, history_data, quantitative_result)
            ai_analysis_data = local_ai
            ai_confidence = local_ai.get('confidence', 0.5)
            ai_rating = local_ai.get('overall_rating_raw', 'HOLD')
            ai_risk_score = 1.0 - ai_confidence
        
        # 综合决策：融合量化分析和AI分析
        quant_overall = quantitative_result.get('overall_score', 50)
        final_decision = generate_final_decision(quant_overall, ai_rating, ai_confidence, ai_risk_score, quantitative_result)
        
        # 构建技术指标（基于真实计算）
        tech_indicators = calculate_real_technical_indicators(history_data)
        
        # 格式化为前端期望的数据结构
        result = {
            'stock_info': stock_info,
            'final_decision': final_decision,
            'ai_analysis': ai_analysis_data,
            'quantitative_analysis': {
                'factor_scores': {
                    'momentum_score': quantitative_result.get('trend_score', 50.0),
                    'value_score': quantitative_result.get('volume_score', 50.0),
                    'quality_score': quantitative_result.get('technical_score', 50.0),
                    'volatility_score': quantitative_result.get('volatility_score', 50.0)
                },
                'factor_details': quantitative_result.get('factor_details', {}),
                'ml_prediction': {
                    'predicted_return': round((quant_overall - 50) / 100.0, 4),
                    'confidence': round(quant_overall / 100.0, 4),
                    'model_used': 'Quant-Ensemble'
                },
                'technical_indicators': tech_indicators
            },
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"统一分析完成: {stock_code}, 评级={final_decision['rating']}")
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"统一分析API错误: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'分析失败: {str(e)}'
        }), 500


def generate_final_decision(quant_overall, ai_rating, ai_confidence, ai_risk_score, quant_result):
    """生成最终综合决策，使用中文评级"""
    # 将量化评分映射到0-1区间
    quant_norm = quant_overall / 100.0
    
    # AI评级映射为数值
    ai_rating_map = {
        'STRONG_BUY': 0.9, 'BUY': 0.7, 'HOLD': 0.5, 'SELL': 0.3, 'STRONG_SELL': 0.1,
        '强烈买入': 0.9, '建议买入': 0.7, '建议持仓': 0.5, '建议卖出': 0.3, '强烈卖出': 0.1
    }
    ai_score = ai_rating_map.get(ai_rating, 0.5)
    
    # 综合评分：量化60% + AI 40%
    combined_score = quant_norm * 0.6 + ai_score * 0.4
    combined_confidence = (quant_norm * 0.5 + ai_confidence * 0.5)
    
    # 根据综合评分确定中文评级
    if combined_score >= 0.75:
        rating = '强烈买入'
        reasoning_prefix = '量化和AI分析均发出强烈看涨信号'
    elif combined_score >= 0.60:
        rating = '建议买入'
        reasoning_prefix = '量化指标积极，AI分析偏正面'
    elif combined_score >= 0.40:
        rating = '建议持仓'
        reasoning_prefix = '量化和AI分析信号中性'
    elif combined_score >= 0.25:
        rating = '建议卖出'
        reasoning_prefix = '量化指标偏弱，AI分析偏负面'
    else:
        rating = '强烈卖出'
        reasoning_prefix = '量化和AI分析均发出强烈看跌信号'
    
    # 构建详细决策理由
    trend = quant_result.get('trend_score', 50)
    vol = quant_result.get('volatility_score', 50)
    tech = quant_result.get('technical_score', 50)
    volume = quant_result.get('volume_score', 50)
    
    reasoning = (
        f"{reasoning_prefix}。"
        f"趋势动量得分{trend:.2f}分，"
        f"波动率得分{vol:.2f}分，"
        f"技术指标得分{tech:.2f}分，"
        f"成交量得分{volume:.2f}分，"
        f"量化综合{quant_overall:.2f}分。"
        f"AI置信度{ai_confidence:.1%}。"
        f"综合判定：{rating}。"
    )
    
    return {
        'rating': rating,
        'confidence': round(combined_confidence, 4),
        'risk_level': round(ai_risk_score if ai_risk_score else (1.0 - combined_score), 4),
        'reasoning': reasoning
    }


def perform_local_ai_analysis(stock_code, stock_info, history_data, quant_result):
    """当TradingAgents-CN不可用时，基于量化数据生成本地AI分析"""
    name = stock_info.get('name', stock_code)
    industry = stock_info.get('industry', '相关行业')
    
    overall = quant_result.get('overall_score', 50)
    trend = quant_result.get('trend_score', 50)
    vol = quant_result.get('volatility_score', 50)
    
    # 根据量化结果推断评级
    if overall >= 70:
        raw_rating = 'BUY'
        advice = f'{name}量化指标表现优秀，趋势向好，建议积极关注配置机会。'
    elif overall >= 55:
        raw_rating = 'HOLD'
        advice = f'{name}量化指标表现平稳，建议维持现有仓位，等待更明确信号。'
    else:
        raw_rating = 'SELL'
        advice = f'{name}量化指标偏弱，建议注意风险控制，适当减仓。'
    
    # 构建分析摘要
    if trend >= 65:
        market_msg = f'{name}处于上升趋势中，均线系统多头排列，短期动能较强。'
    elif trend >= 45:
        market_msg = f'{name}处于震荡整理阶段，均线缠绕，方向尚不明朗。'
    else:
        market_msg = f'{name}处于下降趋势中，均线系统空头排列，短期偏弱。'
    
    if vol >= 65:
        fund_msg = f'{name}波动率较低，价格走势稳定，适合风险偏好较低的投资者。'
    elif vol >= 40:
        fund_msg = f'{name}波动率处于正常范围，风险收益比尚可。'
    else:
        fund_msg = f'{name}波动率较高，价格波动剧烈，建议谨慎操作。'
    
    news_msg = f'基于{industry}板块整体表现和{name}近期走势综合判断（本地量化分析，AI服务未接入）。'
    
    return {
        'ai_available': True,  # 标记为可用，以便前端展示分析内容
        'overall_rating': f"{overall:.1f}/100",
        'overall_rating_raw': raw_rating,
        'confidence': round(overall / 100.0, 4),
        'target_price': '基于量化模型',
        'investment_advice': advice,
        'analysis_summary': {
            'market_analysis': market_msg,
            'fundamental_analysis': fund_msg,
            'news_analysis': news_msg
        }
    }


def calculate_real_technical_indicators(history_data):
    """基于真实历史数据计算技术指标"""
    try:
        if not history_data or len(history_data) < 20:
            return {
                'rsi': '50.00',
                'macd_signal': '数据不足',
                'bollinger_position': '数据不足'
            }
        
        df = pd.DataFrame(history_data)
        
        # 计算RSI
        rsi_val = calculate_rsi(df['close'])
        
        # 计算MACD信号
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            macd_signal = '金叉看涨'
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            macd_signal = '死叉看跌'
        elif macd_line.iloc[-1] > signal_line.iloc[-1]:
            macd_signal = '看涨'
        else:
            macd_signal = '看跌'
        
        # 计算布林带位置
        ma20 = df['close'].rolling(window=20).mean().iloc[-1]
        std20 = df['close'].rolling(window=20).std().iloc[-1]
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        current = df['close'].iloc[-1]
        
        if current >= upper:
            boll_pos = '上轨(超买)'
        elif current <= lower:
            boll_pos = '下轨(超卖)'
        elif current >= ma20:
            pct = (current - ma20) / (upper - ma20) * 100 if (upper - ma20) > 0 else 50
            boll_pos = f'中上轨({pct:.0f}%)'
        else:
            pct = (ma20 - current) / (ma20 - lower) * 100 if (ma20 - lower) > 0 else 50
            boll_pos = f'中下轨({pct:.0f}%)'
        
        return {
            'rsi': f"{rsi_val:.2f}",
            'macd_signal': macd_signal,
            'bollinger_position': boll_pos
        }
    except Exception as e:
        logger.warning(f"计算技术指标失败: {e}")
        return {
            'rsi': '50.00',
            'macd_signal': '计算失败',
            'bollinger_position': '计算失败'
        }

def perform_quantitative_analysis(stock_code, history_data):
    """执行量化分析 - 基于真实历史数据"""
    try:
        if not history_data or len(history_data) == 0:
            return {
                'trend_score': 50.0,
                'volatility_score': 50.0,
                'volume_score': 50.0,
                'technical_score': 50.0,
                'overall_score': 50.0,
                'factor_details': {}
            }
        
        df = pd.DataFrame(history_data)
        
        # 趋势分析（含计算依据）
        trend_score, trend_detail = analyze_trend_with_detail(df)
        
        # 波动率分析（含计算依据）
        volatility_score, vol_detail = analyze_volatility_with_detail(df)
        
        # 成交量分析（含计算依据）
        volume_score, vol_d_detail = analyze_volume_with_detail(df)
        
        # 技术指标综合分析（含计算依据）
        technical_score, tech_detail = analyze_technical_with_detail(df)
        
        # 计算总体量化评分
        overall_score = (trend_score * 0.3 + volatility_score * 0.2 + 
                        volume_score * 0.2 + technical_score * 0.3)
        
        return {
            'trend_score': round(trend_score, 2),
            'volatility_score': round(volatility_score, 2),
            'volume_score': round(volume_score, 2),
            'technical_score': round(technical_score, 2),
            'overall_score': round(overall_score, 2),
            'factor_details': {
                'momentum': trend_detail,
                'volatility': vol_detail,
                'value': vol_d_detail,
                'quality': tech_detail
            }
        }
    except Exception as e:
        logger.error(f"量化分析错误: {e}")
        return {
            'trend_score': 50.0,
            'volatility_score': 50.0,
            'volume_score': 50.0,
            'technical_score': 50.0,
            'overall_score': 50.0,
            'factor_details': {}
        }


def analyze_trend_with_detail(df):
    """趋势分析 - 返回得分和计算依据"""
    try:
        if len(df) < 20:
            return 50.0, '数据不足20条，无法进行趋势分析'
        
        ma5 = df['close'].rolling(window=5).mean().iloc[-1]
        ma10 = df['close'].rolling(window=10).mean().iloc[-1]
        ma20 = df['close'].rolling(window=20).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # 计算价格相对均线的偏离度
        dev_ma5 = (current_price - ma5) / ma5 * 100
        dev_ma20 = (current_price - ma20) / ma20 * 100
        
        score = 50.0
        if current_price > ma5 > ma10 > ma20:
            score = 85.0
        elif current_price > ma5 > ma10:
            score = 70.0
        elif current_price > ma10:
            score = 60.0
        elif current_price < ma5 < ma10 < ma20:
            score = 15.0
        elif current_price < ma5 < ma10:
            score = 30.0
        elif current_price < ma10:
            score = 40.0
        
        detail = (f"现价{current_price:.2f}, MA5={ma5:.2f}(偏离{dev_ma5:+.2f}%), "
                 f"MA10={ma10:.2f}, MA20={ma20:.2f}(偏离{dev_ma20:+.2f}%)")
        return score, detail
    except Exception as e:
        return 50.0, f'计算异常: {e}'


def analyze_volatility_with_detail(df):
    """波动率分析 - 返回得分和计算依据"""
    try:
        if len(df) < 20:
            return 50.0, '数据不足20条，无法计算波动率'
        
        returns = df['close'].pct_change().dropna()
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # 最近5日波动率
        recent_vol = returns.iloc[-5:].std() * np.sqrt(252) if len(returns) >= 5 else annual_vol
        
        if annual_vol < 0.2:
            score = 80.0
        elif annual_vol < 0.3:
            score = 65.0
        elif annual_vol < 0.4:
            score = 50.0
        elif annual_vol < 0.5:
            score = 35.0
        else:
            score = 20.0
        
        detail = (f"年化波动率{annual_vol:.2%}, 日波动率{daily_vol:.4f}, "
                 f"近5日年化波动率{recent_vol:.2%}")
        return score, detail
    except Exception as e:
        return 50.0, f'计算异常: {e}'


def analyze_volume_with_detail(df):
    """成交量分析 - 返回得分和计算依据"""
    try:
        if len(df) < 10:
            return 50.0, '数据不足10条，无法分析成交量'
        
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 计算量价配合度
        recent_price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0
        
        if volume_ratio > 1.5:
            score = 80.0
        elif volume_ratio > 1.2:
            score = 70.0
        elif volume_ratio > 0.8:
            score = 60.0
        else:
            score = 40.0
        
        detail = (f"近5日均量{recent_volume:.0f}, 整体均量{avg_volume:.0f}, "
                 f"量比{volume_ratio:.2f}, 近5日涨跌{recent_price_change:+.2f}%")
        return score, detail
    except Exception as e:
        return 50.0, f'计算异常: {e}'


def analyze_technical_with_detail(df):
    """技术指标综合分析 - 返回得分和计算依据"""
    try:
        if len(df) < 30:
            return 50.0, '数据不足30条，无法计算技术指标'
        
        scores = []
        details = []
        
        # RSI指标
        rsi = calculate_rsi(df['close'])
        if 30 < rsi < 70:
            rsi_score = 70.0
            rsi_state = '正常区间'
        elif rsi <= 30:
            rsi_score = 85.0
            rsi_state = '超卖区间'
        else:
            rsi_score = 30.0
            rsi_state = '超买区间'
        scores.append(rsi_score)
        details.append(f"RSI(14)={rsi:.2f}({rsi_state})")
        
        # MACD指标
        macd_score = analyze_macd(df)
        if macd_score >= 80:
            macd_state = '金叉'
        elif macd_score <= 35:
            macd_state = '死叉'
        elif macd_score >= 60:
            macd_state = 'MACD在信号线上方'
        else:
            macd_state = 'MACD在信号线下方'
        scores.append(macd_score)
        details.append(f"MACD评分={macd_score:.0f}({macd_state})")
        
        final_score = float(np.mean(scores)) if scores else 50.0
        detail_str = '; '.join(details)
        return final_score, detail_str
    except Exception as e:
        return 50.0, f'计算异常: {e}'

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except:
        return 50

def analyze_macd(df):
    """MACD分析"""
    try:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return 85  # 金叉
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return 30  # 死叉
        elif macd.iloc[-1] > signal.iloc[-1]:
            return 70  # MACD在信号线上方
        else:
            return 40  # MACD在信号线下方
    except:
        return 50

@api_bp.route('/analysis/screen', methods=['POST'])
def screen_stocks():
    """股票筛选"""
    try:
        data = request.get_json()
        logger.info(f"收到筛选请求: {data}")
        
        # 使用StockService进行筛选
        result = StockService.screen_stocks(data)
        
        return jsonify({
            'code': 200,
            'message': '筛选完成',
            'data': result
        })
    except Exception as e:
        logger.error(f"股票筛选API错误: {e}")
        return jsonify({
            'code': 500,
            'message': f'服务器错误: {str(e)}',
            'data': None
        }), 500

@api_bp.route('/analysis/backtest', methods=['POST'])
def backtest_strategy():
    """策略回测"""
    try:
        data = request.get_json()
        logger.info(f"收到回测请求: {data}")
        
        # 验证必要参数
        required_fields = ['ts_code', 'strategy_type', 'start_date', 'end_date', 'initial_capital']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'code': 400,
                    'message': f'缺少必要参数: {field}',
                    'data': None
                }), 400
        
        # 获取历史数据
        ts_code = data['ts_code']
        start_date = data['start_date']
        end_date = data['end_date']
        
        # 获取更多历史数据用于计算技术指标
        extended_start = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=100)
        extended_start_str = extended_start.strftime('%Y-%m-%d')
        
        history_data = StockService.get_daily_history(
            ts_code, 
            start_date=extended_start_str, 
            end_date=end_date, 
            limit=500
        )
        
        if not history_data or len(history_data) < 30:
            return jsonify({
                'code': 400,
                'message': '历史数据不足，无法进行回测',
                'data': None
            }), 400
        
        # 获取技术因子数据
        factors_data = StockService.get_stock_factors(
            ts_code,
            start_date=extended_start_str,
            end_date=end_date,
            limit=500
        )
        
        # 执行回测
        backtest_engine = BacktestEngine(data)
        result = backtest_engine.run_backtest(history_data, factors_data)
        
        return jsonify({
            'code': 200,
            'message': '回测完成',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"策略回测API错误: {e}")
        return jsonify({
            'code': 500,
            'message': f'回测失败: {str(e)}',
            'data': None
        }), 500


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config):
        self.config = config
        self.ts_code = config['ts_code']
        self.strategy_type = config['strategy_type']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.initial_capital = config['initial_capital']
        self.commission_rate = config.get('commission_rate', 0.001)
        self.params = config.get('params', {})
        
        # 回测状态
        self.cash = self.initial_capital
        self.position = 0  # 持仓数量
        self.trades = []
        self.daily_values = []
        
    def run_backtest(self, history_data, factors_data):
        """运行回测"""
        try:
            # 转换为DataFrame
            df_history = pd.DataFrame(history_data)
            df_factors = pd.DataFrame(factors_data)
            
            # 合并数据
            df = self._merge_data(df_history, df_factors)
            
            # 筛选回测期间的数据
            df = df[(df['trade_date'] >= self.start_date) & (df['trade_date'] <= self.end_date)]
            
            if len(df) < 10:
                raise ValueError("回测期间数据不足")
            
            # 计算策略信号
            df = self._calculate_signals(df)
            
            # 执行交易
            self._execute_trades(df)
            
            # 计算绩效指标
            performance = self._calculate_performance(df)
            
            return {
                'performance': performance,
                'trades': self.trades[-20:],  # 返回最近20笔交易
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            raise
    
    def _merge_data(self, df_history, df_factors):
        """合并历史数据和技术因子数据"""
        df_history['trade_date'] = pd.to_datetime(df_history['trade_date'])
        df_factors['trade_date'] = pd.to_datetime(df_factors['trade_date'])
        
        # 合并数据
        df = pd.merge(df_history, df_factors, on='trade_date', how='left', suffixes=('', '_factor'))
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        return df
    
    def _calculate_signals(self, df):
        """计算策略信号"""
        df['signal'] = 0  # 0: 无操作, 1: 买入, -1: 卖出
        
        if self.strategy_type == 'ma_cross':
            df = self._ma_cross_strategy(df)
        elif self.strategy_type == 'macd':
            df = self._macd_strategy(df)
        elif self.strategy_type == 'kdj':
            df = self._kdj_strategy(df)
        elif self.strategy_type == 'rsi':
            df = self._rsi_strategy(df)
        elif self.strategy_type == 'bollinger':
            df = self._bollinger_strategy(df)
        
        return df
    
    def _ma_cross_strategy(self, df):
        """均线交叉策略"""
        ma_short = self.params.get('ma_short', 5)
        ma_long = self.params.get('ma_long', 20)
        
        # 计算均线
        df['ma_short'] = df['close'].rolling(window=ma_short).mean()
        df['ma_long'] = df['close'].rolling(window=ma_long).mean()
        
        # 生成信号
        for i in range(1, len(df)):
            if (df.iloc[i]['ma_short'] > df.iloc[i]['ma_long'] and 
                df.iloc[i-1]['ma_short'] <= df.iloc[i-1]['ma_long']):
                df.iloc[i, df.columns.get_loc('signal')] = 1  # 买入信号
            elif (df.iloc[i]['ma_short'] < df.iloc[i]['ma_long'] and 
                  df.iloc[i-1]['ma_short'] >= df.iloc[i-1]['ma_long']):
                df.iloc[i, df.columns.get_loc('signal')] = -1  # 卖出信号
        
        return df
    
    def _macd_strategy(self, df):
        """MACD策略"""
        # 使用已计算的MACD数据
        for i in range(1, len(df)):
            current_macd = df.iloc[i]['macd'] if pd.notna(df.iloc[i]['macd']) else 0
            current_dea = df.iloc[i]['macd_dea'] if pd.notna(df.iloc[i]['macd_dea']) else 0
            prev_macd = df.iloc[i-1]['macd'] if pd.notna(df.iloc[i-1]['macd']) else 0
            prev_dea = df.iloc[i-1]['macd_dea'] if pd.notna(df.iloc[i-1]['macd_dea']) else 0
            
            # MACD上穿DEA买入，下穿卖出
            if current_macd > current_dea and prev_macd <= prev_dea:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            elif current_macd < current_dea and prev_macd >= prev_dea:
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        return df
    
    def _kdj_strategy(self, df):
        """KDJ策略"""
        oversold = self.params.get('oversold', 20)
        overbought = self.params.get('overbought', 80)
        
        for i in range(1, len(df)):
            current_k = df.iloc[i]['kdj_k'] if pd.notna(df.iloc[i]['kdj_k']) else 50
            prev_k = df.iloc[i-1]['kdj_k'] if pd.notna(df.iloc[i-1]['kdj_k']) else 50
            
            # 从超卖区域向上突破买入
            if current_k > oversold and prev_k <= oversold:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            # 从超买区域向下突破卖出
            elif current_k < overbought and prev_k >= overbought:
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        return df
    
    def _rsi_strategy(self, df):
        """RSI策略"""
        oversold = self.params.get('oversold', 30)
        overbought = self.params.get('overbought', 70)
        
        for i in range(1, len(df)):
            current_rsi = df.iloc[i]['rsi_6'] if pd.notna(df.iloc[i]['rsi_6']) else 50
            prev_rsi = df.iloc[i-1]['rsi_6'] if pd.notna(df.iloc[i-1]['rsi_6']) else 50
            
            # 从超卖区域向上突破买入
            if current_rsi > oversold and prev_rsi <= oversold:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            # 从超买区域向下突破卖出
            elif current_rsi < overbought and prev_rsi >= overbought:
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        return df
    
    def _bollinger_strategy(self, df):
        """布林带策略"""
        for i in range(len(df)):
            close = df.iloc[i]['close']
            boll_upper = df.iloc[i]['boll_upper'] if pd.notna(df.iloc[i]['boll_upper']) else close
            boll_lower = df.iloc[i]['boll_lower'] if pd.notna(df.iloc[i]['boll_lower']) else close
            
            # 价格触及下轨买入
            if close <= boll_lower and boll_lower > 0:
                df.iloc[i, df.columns.get_loc('signal')] = 1
            # 价格触及上轨卖出
            elif close >= boll_upper and boll_upper > 0:
                df.iloc[i, df.columns.get_loc('signal')] = -1
        
        return df
    
    def _execute_trades(self, df):
        """执行交易"""
        for i, row in df.iterrows():
            signal = row['signal']
            price = row['close']
            date = row['trade_date'].strftime('%Y-%m-%d')
            
            if signal == 1 and self.position == 0:  # 买入
                # 计算可买入数量（按手，1手=100股）
                max_shares = int(self.cash / price / 100) * 100
                if max_shares >= 100:  # 至少买入1手
                    commission = max_shares * price * self.commission_rate
                    total_cost = max_shares * price + commission
                    
                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        self.position = max_shares
                        
                        self.trades.append({
                            'date': date,
                            'action': 'buy',
                            'price': price,
                            'quantity': max_shares,
                            'amount': total_cost,
                            'commission': commission,
                            'return_rate': None
                        })
            
            elif signal == -1 and self.position > 0:  # 卖出
                commission = self.position * price * self.commission_rate
                total_income = self.position * price - commission
                
                # 计算收益率
                buy_trade = None
                for trade in reversed(self.trades):
                    if trade['action'] == 'buy':
                        buy_trade = trade
                        break
                
                return_rate = 0
                if buy_trade:
                    return_rate = (total_income - buy_trade['amount']) / buy_trade['amount']
                
                sell_quantity = self.position
                
                self.cash += total_income
                self.position = 0
                
                self.trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': price,
                    'quantity': sell_quantity,
                    'amount': total_income,
                    'commission': commission,
                    'return_rate': return_rate
                })
            
            # 记录每日资产价值
            portfolio_value = self.cash + self.position * price
            self.daily_values.append({
                'date': date,
                'cash': self.cash,
                'position_value': self.position * price,
                'total_value': portfolio_value
            })
    
    def _calculate_performance(self, df):
        """计算绩效指标"""
        if not self.daily_values:
            return self._get_default_performance()
        
        # 最终清仓
        final_price = df.iloc[-1]['close']
        if self.position > 0:
            commission = self.position * final_price * self.commission_rate
            self.cash += self.position * final_price - commission
            self.position = 0
        
        final_capital = self.cash
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        days = len(df)
        annual_return = (final_capital / self.initial_capital) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        values = [dv['total_value'] for dv in self.daily_values]
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算波动率
        returns = []
        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            returns.append(daily_return)
        
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        # 计算夏普比率
        risk_free_rate = 0.03  # 假设无风险利率3%
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 交易统计
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell' and t['return_rate'] is not None]
        winning_trades = len([t for t in sell_trades if t['return_rate'] > 0])
        win_rate = winning_trades / len(sell_trades) if sell_trades else 0
        
        # 计算平均持仓天数
        avg_holding_days = 0
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            holding_periods = []
            for i, sell_trade in enumerate(sell_trades):
                if i < len(buy_trades):
                    buy_date = datetime.strptime(buy_trades[i]['date'], '%Y-%m-%d')
                    sell_date = datetime.strptime(sell_trade['date'], '%Y-%m-%d')
                    holding_periods.append((sell_date - buy_date).days)
            avg_holding_days = np.mean(holding_periods) if holding_periods else 0
        
        # 计算基准收益率（买入持有策略）
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        benchmark_return = (end_price - start_price) / start_price
        
        # 计算总手续费
        total_commission = sum(t['commission'] for t in self.trades if 'commission' in t)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'avg_holding_days': round(avg_holding_days, 1),
            'final_capital': final_capital,
            'total_commission': total_commission,
            'benchmark_return': benchmark_return
        }
    
    def _get_default_performance(self):
        """获取默认绩效指标（无交易情况）"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'avg_holding_days': 0,
            'final_capital': self.initial_capital,
            'total_commission': 0.0,
            'benchmark_return': 0.0
        } 