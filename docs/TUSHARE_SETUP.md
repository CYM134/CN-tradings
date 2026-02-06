# Tushare Token 配置说明

## 什么是 Tushare？

Tushare 是一个免费的中国金融数据接口，提供：
- 股票实时/历史行情数据
- 财务报表数据（资产负债表、利润表、现金流量表）
- 基本面数据（PE、PB、ROE等）
- 指数、基金、期货等数据

官网：https://tushare.pro/

## 为什么需要 Token？

Tushare 使用 token 进行身份验证，以：
- 控制 API 访问频率
- 提供更稳定的数据服务
- 根据积分等级提供不同权限

**注意**：如果不设置 Tushare token，系统会自动使用 **Baostock** 作为备选数据源，功能正常但数据可能略有差异。

## 如何获取 Token

### 步骤 1：注册账号
1. 访问 https://tushare.pro/register
2. 填写邮箱、用户名、密码完成注册
3. 验证邮箱

### 步骤 2：获取 Token
1. 登录后访问 https://tushare.pro/user/token
2. 在"接口TOKEN"处复制你的 token（类似：`a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`）

### 步骤 3：积分说明
- 新注册用户默认有 **100积分**
- 每日签到可获得积分
- 不同积分等级有不同的数据访问权限
- 基础功能（日线行情、基本面数据）对新用户完全开放

## 如何配置 Token

### 方法一：使用环境变量（推荐）

1. 复制 `.env.example` 为 `.env`：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，添加你的 token：
```bash
# Tushare配置
TUSHARE_TOKEN=你的token字符串
```

### 方法二：直接修改配置文件

编辑 `config_simple.py` 或 `config.py`，修改：
```python
TUSHARE_TOKEN = '你的token字符串'
```

### 方法三：在代码中设置

在需要使用的地方：
```python
import tushare as ts
ts.set_token('你的token字符串')
pro = ts.pro_api()
```

## 验证配置

启动应用后，如果看到：
```
WARNING:app.services.realtime_data_manager:未设置Tushare token，将使用Baostock数据源
```
说明 token 未配置或配置错误。

配置正确后，该警告将消失。

## 常见问题

### Q1: Token 配置了但仍然显示警告？
**A:** 检查：
1. `.env` 文件中的配置格式是否正确（无空格、引号）
2. 重启应用使配置生效
3. Token 是否完整（约32位字符）

### Q2: 提示 "权限不足" 或 "积分不够"？
**A:** 
- 登录 Tushare 账号查看积分
- 每日签到获取积分
- 降低数据请求频率

### Q3: 不配置 Token 能否使用系统？
**A:** 可以！系统会自动使用 Baostock 数据源，基本功能完全正常。

### Q4: Tushare 和 Baostock 有什么区别？
**A:**
- **Tushare**: 数据更全面、更新更及时、提供更多财务数据
- **Baostock**: 免费无需注册、数据略少但够用

## 推荐配置

```bash
# .env 文件示例
TUSHARE_TOKEN=your_actual_token_here
DB_HOST=117.72.178.88
DB_PORT=3306
DB_USER=ai_market
DB_PASSWORD=kGCr8W67S7HMcepR
DB_NAME=ai_market
```

## 数据接口使用示例

```python
import tushare as ts

# 设置token
ts.set_token('your_token')
pro = ts.pro_api()

# 获取股票列表
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

# 获取日线行情
df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240201')

# 获取财务数据
df = pro.income(ts_code='000001.SZ', period='20231231')
```

## 相关链接

- Tushare 官网：https://tushare.pro/
- Tushare 文档：https://tushare.pro/document/2
- 积分获取：https://tushare.pro/document/1?doc_id=13
- Baostock 官网：http://baostock.com/
