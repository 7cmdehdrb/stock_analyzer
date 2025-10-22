from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import io
import traceback
import sqlite3
import os
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from dotenv import load_dotenv
import openai

# .env 파일 로드
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(os.path.dirname(__file__), 'stock_cache.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# AI 분석 rate limiting과 캐싱을 위한 딕셔너리
ai_analysis_cache = {}  # {ip: {"timestamp": datetime, "result": dict}}
ai_analysis_rate_limit = {}  # {ip: last_request_time}

# DB 파일 경로
DB_PATH = os.path.join(os.path.dirname(__file__), "stock_cache.db")

# SQLAlchemy 설정
db = SQLAlchemy(app)


# SQLAlchemy 모델 정의
class StockPriceCache(db.Model):
    __tablename__ = "stock_price_cache"

    ticker = db.Column(db.String(20), primary_key=True)
    date = db.Column(db.String(10), primary_key=True)
    close_price = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f"<StockPrice {self.ticker} {self.date}>"


class SavedPortfolio(db.Model):
    __tablename__ = "saved_portfolios"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(200), nullable=False)
    csv_content = db.Column(db.Text, nullable=False)
    start_date = db.Column(db.String(10))
    benchmark_ticker = db.Column(db.String(20))
    base_currency = db.Column(db.String(3))
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_accessed = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Portfolio {self.name}>"


# Flask-Admin ModelView 정의
class StockPriceCacheAdmin(ModelView):
    column_list = ["ticker", "date", "close_price", "created_at"]
    column_searchable_list = ["ticker"]
    column_sortable_list = ["ticker", "date", "created_at"]
    column_default_sort = [("ticker", False), ("date", True)]
    page_size = 50
    can_export = True
    export_types = ["csv", "xlsx"]


class SavedPortfolioAdmin(ModelView):
    column_list = [
        "id",
        "name",
        "start_date",
        "benchmark_ticker",
        "base_currency",
        "created_at",
        "last_accessed",
    ]
    column_searchable_list = ["name", "benchmark_ticker"]
    column_sortable_list = ["id", "name", "created_at", "last_accessed"]
    column_default_sort = [("last_accessed", True)]
    page_size = 50
    can_export = True
    export_types = ["csv", "xlsx"]
    # CSV 내용은 너무 길어서 리스트에서 제외
    column_exclude_list = ["csv_content"]
    # 상세보기/수정에서는 표시
    form_excluded_columns = []


# Flask-Admin 초기화
admin = Admin(app, name="Portfolio Admin", template_mode="bootstrap3")
admin.add_view(StockPriceCacheAdmin(StockPriceCache, db.session, name="Stock Prices"))
admin.add_view(SavedPortfolioAdmin(SavedPortfolio, db.session, name="Portfolios"))


def init_database():
    """데이터베이스 초기화 및 테이블 생성"""
    # SQLAlchemy로 테이블 생성 (이미 존재하면 무시)
    with app.app_context():
        db.create_all()


def get_cached_prices(ticker, start_date, end_date):
    """DB에서 캐시된 가격 데이터 조회"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()

        query = """
            SELECT date, close_price 
            FROM stock_price_cache 
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date
        """

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        cursor.execute(query, (ticker, start_str, end_str))
        results = cursor.fetchall()

        if not results:
            return None

        # DataFrame으로 변환
        df = pd.DataFrame(results, columns=["date", "close_price"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return df["close_price"]
    except Exception as e:
        return None
    finally:
        if conn:
            conn.close()


def save_prices_to_cache(ticker, price_series):
    """가격 데이터를 DB에 저장"""
    if price_series is None or len(price_series) == 0:
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()

        # 데이터 준비 (NaN 값 제외)
        data_to_insert = []
        for date, price in price_series.items():
            # NaN 값 체크
            if pd.notna(price) and not np.isnan(price):
                date_str = date.strftime("%Y-%m-%d")
                data_to_insert.append((ticker, date_str, float(price)))

        if not data_to_insert:
            return

        # INSERT OR REPLACE로 중복 방지
        cursor.executemany(
            """
            INSERT OR REPLACE INTO stock_price_cache (ticker, date, close_price)
            VALUES (?, ?, ?)
        """,
            data_to_insert,
        )

        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def get_current_exchange_rate():
    """현재 USD/KRW 환율 가져오기"""
    try:
        # USDKRW=X 티커로 환율 정보 가져오기
        krw = yf.Ticker("USDKRW=X")
        data = krw.history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
        else:
            # 기본값 (최근 평균 환율)
            return 1350.0
    except Exception as e:
        return 1350.0


def fetch_stock_data(ticker, start_date, end_date):
    """Yahoo Finance에서 주가 데이터 가져오기 (DB 캐시 활용)"""
    try:
        # 1. 먼저 DB 캐시에서 데이터 조회
        cached_data = get_cached_prices(ticker, start_date, end_date)

        # 2. 캐시에 모든 데이터가 있는지 확인
        if cached_data is not None and len(cached_data) > 0:
            # 날짜 범위 확인
            expected_start = pd.to_datetime(start_date)
            expected_end = pd.to_datetime(end_date)
            cached_start = cached_data.index.min()
            cached_end = cached_data.index.max()

            # 캐시가 요청 범위를 모두 커버하는지 확인 (±7일 허용)
            if cached_start <= expected_start + timedelta(
                days=7
            ) and cached_end >= expected_end - timedelta(days=7):
                return cached_data

        # 3. 캐시에 없으면 Yahoo Finance에서 가져오기
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            return None

        price_data = data["Close"]

        # timezone 제거 (캐시된 데이터와 일관성 유지)
        if hasattr(price_data.index, 'tz') and price_data.index.tz is not None:
            price_data.index = price_data.index.tz_localize(None)

        # NaN 값 제거
        price_data = price_data.dropna()

        if len(price_data) == 0:
            return None

        # 4. 새로 가져온 데이터를 DB에 저장
        save_prices_to_cache(ticker, price_data)

        return price_data

    except Exception as e:
        return None


def normalize_ticker(ticker, country):
    """티커 정규화 (한국 종목에 .KS 자동 추가)"""
    ticker = str(ticker).strip()

    # 한국 종목인 경우
    if country == "한국":
        # 숫자로만 이루어진 경우 (예: 005930)
        if ticker.isdigit():
            ticker = f"{ticker}.KS"
            return ticker
        # 이미 .KS나 .KQ가 붙어있지 않은 경우
        elif not (ticker.endswith(".KS") or ticker.endswith(".KQ")):
            ticker = f"{ticker}.KS"
            return ticker

    return ticker


def merge_duplicate_tickers(portfolio_df):
    """동일 티커를 가진 종목의 보유량을 합산"""
    # 필수 컬럼 확인
    if "티커" not in portfolio_df.columns or "보유량" not in portfolio_df.columns:
        return portfolio_df

    # 티커 정규화 (한국 종목에 .KS 추가)
    if "국가" in portfolio_df.columns:
        portfolio_df["티커"] = portfolio_df.apply(
            lambda row: normalize_ticker(row["티커"], row.get("국가", "")), axis=1
        )

    # 티커별로 그룹화하여 보유량 합산
    # 첫 번째 행의 다른 정보(종목명, 국가, 분류 등)는 유지
    grouped = portfolio_df.groupby("티커", as_index=False).agg(
        {
            "보유량": "sum",  # 보유량은 합산
            **{
                col: "first"
                for col in portfolio_df.columns
                if col not in ["티커", "보유량"]
            },
        }
    )

    return grouped


def fill_missing_dates(price_series, start_date, end_date):
    """휴장일 및 상장일로 인한 빈 데이터 처리

    Args:
        price_series: 주가 시계열 데이터
        start_date: 분석 시작 날짜
        end_date: 분석 종료 날짜

    Returns:
        보간된 주가 시계열 데이터
    """
    if price_series is None or len(price_series) == 0:
        print("  ⚠ fill_missing_dates: No data provided")
        return None

    try:
        print(f"  📅 Date range: {start_date.date()} to {end_date.date()}")
        print(f"  📊 Original data: {len(price_series)} trading days")
        print(f"  🔍 Index type: {type(price_series.index)}, dtype: {price_series.index.dtype}")
        
        # 원본 데이터의 인덱스를 timezone-naive로 변환
        if hasattr(price_series.index, 'tz') and price_series.index.tz is not None:
            print(f"  🌐 Converting from timezone: {price_series.index.tz}")
            price_series.index = price_series.index.tz_localize(None)
        
        # 인덱스가 DatetimeIndex인지 확인
        if not isinstance(price_series.index, pd.DatetimeIndex):
            print(f"  🔄 Converting index to DatetimeIndex")
            price_series.index = pd.to_datetime(price_series.index)
        
        # 전체 날짜 범위 생성 (모든 날짜 포함)
        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        print(f"  � Expanded range: {len(all_dates)} days")
        print(f"  📊 First original date: {price_series.index[0]}, Last: {price_series.index[-1]}")

        # 기존 데이터를 전체 날짜 범위로 확장
        price_series_filled = price_series.reindex(all_dates)
        
        # 데이터가 있는 첫 날짜 확인 (상장일)
        first_valid_date = price_series_filled.first_valid_index()

        if first_valid_date is None:
            print("  ❌ No valid dates found after reindex")
            print(f"  🔍 Checking overlap: orig min={price_series.index.min()}, max={price_series.index.max()}")
            print(f"  🔍 Checking overlap: new min={all_dates.min()}, max={all_dates.max()}")
            # 그냥 원본 데이터 반환 (날짜 확장 없이)
            return price_series

        # 상장 이전 데이터: 상장 후 첫 가격으로 채움
        first_price = price_series_filled[first_valid_date]
        price_series_filled.loc[:first_valid_date] = first_price

        # 상장 이후 빈 데이터: 선형 보간
        price_series_filled = price_series_filled.interpolate(method="linear", limit_direction="forward")

        # 아직도 빈 값이 있다면 (끝 부분) forward fill
        price_series_filled = price_series_filled.ffill()
        
        # 그래도 남아있는 NaN은 backward fill
        price_series_filled = price_series_filled.bfill()

        filled_count = len(all_dates) - len(price_series)
        print(f"  ✓ Filled {filled_count} missing dates (total: {len(price_series_filled)} days)")

        return price_series_filled
        
    except Exception as e:
        print(f"  ❌ Error in fill_missing_dates: {e}")
        import traceback
        traceback.print_exc()
        # 오류 시 원본 데이터 반환
        return price_series


def calculate_portfolio_returns(portfolio_df, start_date, base_currency="USD"):
    """포트폴리오 수익률 계산 (기준 통화 적용, 현금 제외)"""
    end_date = datetime.now()

    # 현재 환율 가져오기
    exchange_rate = get_current_exchange_rate()
    print(f"Current USD/KRW exchange rate: {exchange_rate}")

    # 각 종목의 수익률 데이터 수집
    portfolio_data = {}
    cash_holdings = {}
    failed_tickers = []  # 실패한 티커 추적
    total_initial_value = 0
    total_initial_value_with_cash = 0

    for _, row in portfolio_df.iterrows():
        ticker = row["티커"]
        quantity = row["보유량"]
        country = row.get("국가", "미국")  # 기본값은 미국
        asset_class = row.get("분류", "")

        # 현금인 경우 별도 처리
        if asset_class == "현금":
            # 현금 가치 계산 (환율 적용)
            if country == "한국" and base_currency == "USD":
                cash_value = quantity / exchange_rate
            elif country == "미국" and base_currency == "KRW":
                cash_value = quantity * exchange_rate
            else:
                cash_value = quantity

            cash_holdings[ticker] = {
                "value": cash_value,
                "country": country,
                "ticker": ticker,
            }
            total_initial_value_with_cash += cash_value
            continue

        # 주가 데이터 가져오기
        price_data = fetch_stock_data(ticker, start_date, end_date)

        if price_data is None or len(price_data) == 0:
            print(f"⚠ Skipping {ticker}: No price data available")
            failed_tickers.append(ticker)  # 실패한 티커 기록
            continue

        # 환율 적용
        # 기준 통화가 USD이고 한국 주식인 경우 -> USD로 환산
        # 기준 통화가 KRW이고 미국 주식인 경우 -> KRW로 환산
        if base_currency == "USD" and country == "한국":
            # 한국 주식을 USD로 환산 (KRW / 환율)
            price_data = price_data / exchange_rate
        elif base_currency == "KRW" and country == "미국":
            # 미국 주식을 KRW로 환산 (USD * 환율)
            price_data = price_data * exchange_rate

        initial_price = price_data.iloc[0]
        initial_value = initial_price * quantity
        total_initial_value += initial_value
        total_initial_value_with_cash += initial_value

        portfolio_data[ticker] = {
            "prices": price_data,
            "quantity": quantity,
            "initial_value": initial_value,
            "country": country,
            "asset_class": asset_class,
            "name": row.get("종목명", ticker),
        }
        print(f"✓ Added {ticker} to portfolio")

    if not portfolio_data:
        print(f"❌ No valid portfolio data found. Cash holdings: {len(cash_holdings)}")
        return None, None, None, cash_holdings, total_initial_value_with_cash, failed_tickers

    # 모든 날짜의 합집합 구하기 (start_date 이후만)
    all_dates = pd.DatetimeIndex([])
    
    # start_date를 timezone-naive로 변환 (fetch_stock_data가 timezone 없는 데이터 반환)
    start_date_tz = pd.to_datetime(start_date)
    
    for data in portfolio_data.values():
        prices_index = data["prices"].index
        
        # start_date 이후 데이터만 사용
        ticker_dates = prices_index[prices_index >= start_date_tz]
        all_dates = all_dates.union(ticker_dates)

    all_dates = sorted(all_dates)
    
    print(f"\n📅 Portfolio date range:")
    print(f"  Start date requested: {start_date.date()}")
    print(f"  Actual start date: {all_dates[0].date() if len(all_dates) > 0 else 'N/A'}")
    print(f"  End date: {all_dates[-1].date() if len(all_dates) > 0 else 'N/A'}")
    print(f"  Total trading days: {len(all_dates)}")

    # 포트폴리오 전체 가치 계산
    portfolio_values = []

    for date in all_dates:
        daily_value = 0
        for ticker, data in portfolio_data.items():
            # 해당 날짜의 가격 (없으면 forward fill)
            if date in data["prices"].index:
                price = data["prices"][date]
            else:
                # 가장 최근 가격 사용
                available_prices = data["prices"][data["prices"].index <= date]
                if len(available_prices) > 0:
                    price = available_prices.iloc[-1]
                else:
                    price = data["prices"].iloc[0]

            daily_value += price * data["quantity"]

        portfolio_values.append(daily_value)

    portfolio_series = pd.Series(portfolio_values, index=all_dates)
    
    print(f"\n💰 Portfolio values:")
    print(f"  Initial value: ${portfolio_series.iloc[0]:,.2f}")
    print(f"  Final value: ${portfolio_series.iloc[-1]:,.2f}")
    print(f"  Total return: {(portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1) * 100:.2f}%")

    # 일일 수익률 계산
    returns = portfolio_series.pct_change().dropna()
    
    print(f"\n📊 Returns statistics:")
    print(f"  Number of returns: {len(returns)}")
    print(f"  Mean daily return: {returns.mean():.6f} ({returns.mean() * 252 * 100:.2f}% annualized)")
    print(f"  Std daily return: {returns.std():.6f}")
    print(f"  Annualized volatility (√252): {returns.std() * np.sqrt(252) * 100:.2f}%")
    print(f"  Annualized volatility (√trading days): {returns.std() * np.sqrt(len(returns)) * 100:.2f}%")
    print(f"  Min daily return: {returns.min():.4f}")
    print(f"  Max daily return: {returns.max():.4f}")

    # 포트폴리오 데이터와 현금 보유 정보 반환
    return (
        returns,
        portfolio_series,
        portfolio_data,
        cash_holdings,
        total_initial_value_with_cash,
        failed_tickers,  # 실패한 티커 리스트 반환
    )


def calculate_weighted_annual_return(portfolio_returns):
    """연 평균 수익률 계산 (영업일 가중평균)"""
    if len(portfolio_returns) == 0:
        return 0
    
    # 날짜를 연도별로 그룹화
    returns_by_year = {}
    
    for date, ret in portfolio_returns.items():
        year = date.year
        if year not in returns_by_year:
            returns_by_year[year] = []
        returns_by_year[year].append(ret)
    
    # 각 연도의 수익률과 영업일 수 계산
    yearly_data = []
    for year, returns_list in returns_by_year.items():
        trading_days = len(returns_list)
        # 해당 연도의 누적 수익률
        year_cumulative = (1 + pd.Series(returns_list)).prod() - 1
        # 연율화 (해당 연도의 일부만 있는 경우 보정)
        year_return = ((1 + year_cumulative) ** (252 / trading_days) - 1) if trading_days > 0 else 0
        yearly_data.append({
            'year': year,
            'return': year_return,
            'trading_days': trading_days
        })
    
    # 영업일 가중 평균
    total_trading_days = sum(d['trading_days'] for d in yearly_data)
    if total_trading_days == 0:
        return 0
    
    weighted_return = sum(d['return'] * d['trading_days'] for d in yearly_data) / total_trading_days
    
    print(f"\n  📅 Yearly returns (weighted by trading days):")
    for d in yearly_data:
        weight = d['trading_days'] / total_trading_days * 100
        print(f"    {d['year']}: {d['return']*100:.2f}% (weight: {weight:.1f}%, {d['trading_days']} days)")
    print(f"  Weighted average: {weighted_return*100:.2f}%")
    
    return weighted_return


def calculate_metrics(portfolio_returns, benchmark_returns):
    """샤프비, 소티노비, 알파, 베타, 평균 연 수익률 계산"""

    # 인덱스 정보 출력
    print(f"\n🔍 Index comparison:")
    print(f"  Portfolio index type: {type(portfolio_returns.index)}")
    print(f"  Portfolio index tz: {getattr(portfolio_returns.index, 'tz', 'N/A')}")
    print(f"  Portfolio first date: {portfolio_returns.index[0]}")
    print(f"  Benchmark index type: {type(benchmark_returns.index)}")
    print(f"  Benchmark index tz: {getattr(benchmark_returns.index, 'tz', 'N/A')}")
    print(f"  Benchmark first date: {benchmark_returns.index[0]}")

    # timezone 통일
    if hasattr(portfolio_returns.index, 'tz') and portfolio_returns.index.tz is not None:
        portfolio_returns.index = portfolio_returns.index.tz_localize(None)
        print(f"  ✓ Removed timezone from portfolio index")
    
    if hasattr(benchmark_returns.index, 'tz') and benchmark_returns.index.tz is not None:
        benchmark_returns.index = benchmark_returns.index.tz_localize(None)
        print(f"  ✓ Removed timezone from benchmark index")

    # 공통 날짜만 사용
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    print(f"  Common dates found: {len(common_dates)}")
    
    portfolio_returns = portfolio_returns[common_dates]
    benchmark_returns = benchmark_returns[common_dates]

    if len(portfolio_returns) == 0:
        print(f"  ❌ No common dates found!")
        return None

    print(f"\n📊 Calculating metrics:")
    print(f"  Common dates: {len(common_dates)}")
    print(f"  First common date: {common_dates[0].date()}")
    print(f"  Last common date: {common_dates[-1].date()}")
    print(f"  Days in period: {(common_dates[-1] - common_dates[0]).days} calendar days")
    print(f"  Trading days: {len(common_dates)}")
    print(f"  Years: {len(common_dates) / 252:.2f}")
    print(f"\n  Portfolio returns mean: {portfolio_returns.mean():.8f}")
    print(f"  Benchmark returns mean: {benchmark_returns.mean():.8f}")
    print(f"  Portfolio returns std: {portfolio_returns.std():.8f}")
    print(f"  Benchmark returns std: {benchmark_returns.std():.8f}")
    
    # 수익률 차이 확인
    returns_diff = (portfolio_returns - benchmark_returns).abs().mean()
    print(f"  Average absolute difference: {returns_diff:.10f}")
    
    # 샘플 비교
    print(f"\n  📋 Sample comparison (first 5 days):")
    for i in range(min(5, len(common_dates))):
        date = common_dates[i]
        print(f"    {date.date()}: Portfolio={portfolio_returns[date]:.6f}, Benchmark={benchmark_returns[date]:.6f}")

    # 연간화 계산을 위한 거래일 수
    trading_days = 252

    # 평균 연 수익률
    avg_return = portfolio_returns.mean() * trading_days

    # 표준편차 (연간화)
    std_dev = portfolio_returns.std() * np.sqrt(trading_days)

    # 샤프 비율 (무위험 수익률 0으로 가정)
    sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0
    
    print(f"\n  📈 Sharpe Calculation:")
    print(f"    Annualized return: {avg_return * 100:.2f}%")
    print(f"    Annualized volatility: {std_dev * 100:.2f}%")
    print(f"    Sharpe ratio: {sharpe_ratio:.4f}")

    # 소티노 비율 (하방 표준편차)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_days)
    sortino_ratio = avg_return / downside_std if downside_std != 0 else 0
    
    print(f"\n  📉 Sortino Calculation:")
    print(f"    Downside returns count: {len(downside_returns)}/{len(portfolio_returns)}")
    print(f"    Downside volatility: {downside_std * 100:.2f}%")
    print(f"    Sortino ratio: {sortino_ratio:.4f}")

    # 베타 계산 수정 - 공분산과 분산 모두 일일 수익률 기준
    covariance = portfolio_returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

    print(f"  Covariance: {covariance:.6f}")
    print(f"  Benchmark variance: {benchmark_variance:.6f}")
    print(f"  Beta: {beta:.4f}")

    # 알파 (연간화)
    benchmark_avg_return = benchmark_returns.mean() * trading_days
    alpha = avg_return - (beta * benchmark_avg_return)
    
    print(f"  Portfolio annual return: {avg_return * 100:.2f}%")
    print(f"  Benchmark annual return: {benchmark_avg_return * 100:.2f}%")
    print(f"  Alpha: {alpha * 100:.2f}%")

    # 누적 수익률
    cumulative_return = (1 + portfolio_returns).prod() - 1

    # 연수 계산
    years = len(portfolio_returns) / trading_days

    # 연평균 수익률 (CAGR)
    if years > 0:
        cagr = (1 + cumulative_return) ** (1 / years) - 1
    else:
        cagr = 0

    # 벤치마크 CAGR 계산
    benchmark_cumulative_return = (1 + benchmark_returns).prod() - 1
    if years > 0:
        benchmark_cagr = (1 + benchmark_cumulative_return) ** (1 / years) - 1
    else:
        benchmark_cagr = 0

    # 벤치마크 샤프/소티노 계산
    benchmark_std = benchmark_returns.std() * np.sqrt(trading_days)
    benchmark_sharpe = benchmark_avg_return / benchmark_std if benchmark_std != 0 else 0
    
    benchmark_downside_returns = benchmark_returns[benchmark_returns < 0]
    benchmark_downside_std = benchmark_downside_returns.std() * np.sqrt(trading_days)
    benchmark_sortino = benchmark_avg_return / benchmark_downside_std if benchmark_downside_std != 0 else 0

    print(f"\n  📊 Benchmark metrics:")
    print(f"    Sharpe: {benchmark_sharpe:.4f}")
    print(f"    Sortino: {benchmark_sortino:.4f}")
    print(f"    CAGR: {benchmark_cagr * 100:.2f}%")

    # 연 평균 수익률 (영업일 가중평균) 계산
    weighted_annual_return = calculate_weighted_annual_return(portfolio_returns)

    metrics = {
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "benchmark_sharpe_ratio": round(benchmark_sharpe, 4),
        "benchmark_sortino_ratio": round(benchmark_sortino, 4),
        "alpha": round(alpha * 100, 2),  # 퍼센트로 변환
        "beta": round(beta, 4),
        "annual_return": round(avg_return * 100, 2),  # 퍼센트로 변환
        "weighted_annual_return": round(weighted_annual_return * 100, 2),  # 연 평균 수익률
        "benchmark_annual_return": round(benchmark_cagr * 100, 2),  # 벤치마크 CAGR
        "cagr": round(cagr * 100, 2),  # 퍼센트로 변환
        "cumulative_return": round(cumulative_return * 100, 2),
        "volatility": round(std_dev * 100, 2),
        "benchmark_volatility": round(benchmark_std * 100, 2),
    }

    return metrics


def prepare_chart_data(portfolio_returns, benchmark_returns, portfolio_series):
    """차트 데이터 준비"""
    # timezone 통일 (calculate_metrics와 동일)
    if hasattr(portfolio_returns.index, 'tz') and portfolio_returns.index.tz is not None:
        portfolio_returns.index = portfolio_returns.index.tz_localize(None)
    
    if hasattr(benchmark_returns.index, 'tz') and benchmark_returns.index.tz is not None:
        benchmark_returns.index = benchmark_returns.index.tz_localize(None)
    
    if hasattr(portfolio_series.index, 'tz') and portfolio_series.index.tz is not None:
        portfolio_series.index = portfolio_series.index.tz_localize(None)
    
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)

    # 누적 수익률 계산
    portfolio_cumulative = (1 + portfolio_returns[common_dates]).cumprod()
    benchmark_cumulative = (1 + benchmark_returns[common_dates]).cumprod()

    # 날짜를 문자열로 변환
    dates = [date.strftime("%Y-%m-%d") for date in common_dates]

    chart_data = {
        "dates": dates,
        "portfolio_cumulative": [
            round(val, 4) for val in portfolio_cumulative.tolist()
        ],
        "benchmark_cumulative": [
            round(val, 4) for val in benchmark_cumulative.tolist()
        ],
        "portfolio_values": [
            round(val, 2) for val in portfolio_series[common_dates].tolist()
        ],
    }

    return chart_data


def prepare_allocation_data(
    portfolio_data, cash_holdings, total_initial_value_with_cash
):
    """도넛 차트용 자산 배분 데이터 준비"""

    # 시작 시점 배분
    initial_allocation = {}
    for ticker, data in portfolio_data.items():
        name = data.get("name", ticker)
        initial_allocation[name] = data["initial_value"]

    # 현금 추가
    for ticker, cash_data in cash_holdings.items():
        name = f"현금 ({cash_data['country']})"
        if name in initial_allocation:
            initial_allocation[name] += cash_data["value"]
        else:
            initial_allocation[name] = cash_data["value"]

    # 현재 시점 배분
    current_allocation = {}
    for ticker, data in portfolio_data.items():
        name = data.get("name", ticker)
        current_price = data["prices"].iloc[-1]
        current_value = current_price * data["quantity"]
        current_allocation[name] = current_value

    # 현금 추가 (현금은 가치 변동 없음)
    for ticker, cash_data in cash_holdings.items():
        name = f"현금 ({cash_data['country']})"
        if name in current_allocation:
            current_allocation[name] += cash_data["value"]
        else:
            current_allocation[name] = cash_data["value"]

    # 상위 10개 종목만 표시, 나머지는 "기타"로 묶기
    def get_top_allocations(allocation_dict, top_n=10):
        sorted_items = sorted(allocation_dict.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_items) <= top_n:
            return dict(sorted_items)

        top_items = dict(sorted_items[:top_n])
        others_sum = sum(value for _, value in sorted_items[top_n:])
        if others_sum > 0:
            top_items["기타"] = others_sum

        return top_items

    initial_top = get_top_allocations(initial_allocation)
    current_top = get_top_allocations(current_allocation)

    return {
        "initial": {
            "labels": list(initial_top.keys()),
            "values": [round(v, 2) for v in initial_top.values()],
        },
        "current": {
            "labels": list(current_top.keys()),
            "values": [round(v, 2) for v in current_top.values()],
        },
    }


def prepare_holdings_table(portfolio_data, cash_holdings, base_currency, exchange_rate):
    """보유 종목 테이블 데이터 준비 (현재 시점 기준)"""
    holdings = []

    # 투자 자산
    for ticker, data in portfolio_data.items():
        current_price = data["prices"].iloc[-1]
        quantity = data["quantity"]
        current_value = current_price * quantity
        name = data.get("name", ticker)

        holdings.append(
            {
                "ticker": ticker,
                "name": name,
                "quantity": quantity,
                "current_value": round(current_value, 2),
                "asset_class": data.get("asset_class", ""),
            }
        )

    # 현금
    for ticker, cash_data in cash_holdings.items():
        holdings.append(
            {
                "ticker": ticker,
                "name": f"현금 ({cash_data['country']})",
                "quantity": cash_data["value"],
                "current_value": round(cash_data["value"], 2),
                "asset_class": "현금",
            }
        )

    # 총 가치 계산
    total_value = sum(h["current_value"] for h in holdings)

    # 비중 계산
    for holding in holdings:
        holding["weight"] = (
            round((holding["current_value"] / total_value * 100), 2)
            if total_value > 0
            else 0
        )

    # 현재 가치 기준 정렬
    holdings.sort(key=lambda x: x["current_value"], reverse=True)

    return holdings


@app.route("/")
def index():
    """메인 페이지"""
    return render_template("index.html")


@app.route("/api/cache-stats", methods=["GET"])
def get_cache_stats():
    """캐시 통계 조회"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        # 캐시된 티커 수와 레코드 수 조회
        cursor.execute("""
            SELECT COUNT(DISTINCT ticker) as ticker_count,
                   COUNT(*) as record_count
            FROM stock_price_cache
        """)
        
        result = cursor.fetchone()
        ticker_count = result[0] if result else 0
        record_count = result[1] if result else 0
        
        return jsonify({
            "ticker_count": ticker_count,
            "record_count": record_count
        })
    except Exception as e:
        print(f"Error getting cache stats: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route("/api/save-portfolio", methods=["POST"])
def save_portfolio():
    """포트폴리오 분석 결과 저장"""
    try:
        data = request.json
        
        # 필수 데이터 확인
        required_fields = ["name", "csv_content", "start_date", "benchmark_ticker", "base_currency", 
                          "metrics", "summary", "holdings_table", "allocation_data", "chart_data"]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"필수 데이터 누락: {', '.join(missing_fields)}"}), 400
        
        # JSON으로 변환하여 저장
        import json
        
        portfolio = SavedPortfolio(
            name=data["name"],
            csv_content=data["csv_content"],
            start_date=data["start_date"],
            benchmark_ticker=data["benchmark_ticker"],
            base_currency=data["base_currency"],
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        # 분석 결과를 csv_content에 JSON으로 추가 저장
        full_data = {
            "csv_content": data["csv_content"],
            "metrics": data["metrics"],
            "summary": data["summary"],
            "holdings_table": data["holdings_table"],
            "allocation_data": data["allocation_data"],
            "chart_data": data["chart_data"]
        }
        portfolio.csv_content = json.dumps(full_data, ensure_ascii=False)
        
        db.session.add(portfolio)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "portfolio_id": portfolio.id,
            "message": f"포트폴리오 '{data['name']}'가 저장되었습니다."
        })
        
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        print("=" * 80)
        print("❌ ERROR in /api/save-portfolio endpoint:")
        print(error_trace)
        print("=" * 80)
        return jsonify({"error": f"저장 중 오류가 발생했습니다: {str(e)}"}), 500


@app.route("/ranking")
def ranking():
    """랭킹 페이지"""
    return render_template("ranking.html")


@app.route("/api/rankings", methods=["GET"])
def get_rankings():
    """포트폴리오 랭킹 데이터 조회"""
    try:
        import json
        
        # 벤치마크 필터 파라미터 받기
        benchmark_filter = request.args.get('benchmark', None)
        
        # 벤치마크 매핑 (프론트엔드 표시명 -> 실제 티커들)
        # 여러 가능한 티커를 리스트로 관리
        benchmark_mapping = {
            'S&P500': ['SPY', '^GSPC'],
            'NASDAQ100': ['QQQ', '^NDX'],
            'KODEX200': ['069500.KS']
        }
        
        portfolios = SavedPortfolio.query.all()
        
        if not portfolios:
            return jsonify({
                "cagr": [],
                "sortino": [],
                "sharpe": [],
                "alpha": [],
                "beta": []
            })
        
        # 포트폴리오 데이터 파싱
        portfolio_list = []
        for p in portfolios:
            try:
                data = json.loads(p.csv_content)
                
                # 벤치마크 필터링
                if benchmark_filter and benchmark_filter != '전체':
                    expected_tickers = benchmark_mapping.get(benchmark_filter, [])
                    # benchmark_ticker가 없거나 일치하지 않으면 스킵
                    if not p.benchmark_ticker or p.benchmark_ticker not in expected_tickers:
                        continue
                
                portfolio_list.append({
                    "id": p.id,
                    "name": p.name,
                    "created_at": p.created_at.isoformat(),
                    "benchmark": p.benchmark_ticker,
                    "metrics": data.get("metrics", {})
                })
            except Exception as e:
                print(f"Error parsing portfolio {p.id}: {e}")
                continue
        
        # 각 지표별 Top 5
        cagr_top = sorted(portfolio_list, key=lambda x: x["metrics"].get("cagr", -999999), reverse=True)[:5]
        sortino_top = sorted(portfolio_list, key=lambda x: x["metrics"].get("sortino_ratio", -999999), reverse=True)[:5]
        sharpe_top = sorted(portfolio_list, key=lambda x: x["metrics"].get("sharpe_ratio", -999999), reverse=True)[:5]
        alpha_top = sorted(portfolio_list, key=lambda x: x["metrics"].get("alpha", -999999), reverse=True)[:5]
        
        # 베타는 1.0에 가까운 순
        beta_top = sorted(portfolio_list, key=lambda x: abs(x["metrics"].get("beta", 999999) - 1.0))[:5]
        
        return jsonify({
            "cagr": cagr_top,
            "sortino": sortino_top,
            "sharpe": sharpe_top,
            "alpha": alpha_top,
            "beta": beta_top
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print("=" * 80)
        print("❌ ERROR in /api/rankings endpoint:")
        print(error_trace)
        print("=" * 80)
        return jsonify({"error": str(e)}), 500


@app.route("/portfolio/<int:portfolio_id>")
def view_portfolio(portfolio_id):
    """저장된 포트폴리오 상세보기"""
    try:
        import json
        
        portfolio = SavedPortfolio.query.get_or_404(portfolio_id)
        
        # 마지막 접근 시간 업데이트
        portfolio.last_accessed = datetime.now()
        db.session.commit()
        
        # 데이터 파싱
        data = json.loads(portfolio.csv_content)
        
        # 분석 결과 페이지에 전달
        return render_template("portfolio_view.html", 
                             portfolio=portfolio,
                             data=data)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print("=" * 80)
        print("❌ ERROR in /portfolio/<id> endpoint:")
        print(error_trace)
        print("=" * 80)
        return f"포트폴리오를 불러올 수 없습니다: {str(e)}", 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """포트폴리오 분석"""
    try:
        # CSV 파일 읽기
        if "csv_file" not in request.files:
            return jsonify({"error": "CSV 파일이 업로드되지 않았습니다."}), 400

        file = request.files["csv_file"]

        if file.filename == "":
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

        print(f"📁 Received file: {file.filename}")

        start_date = request.form.get("start_date")
        benchmark_ticker = request.form.get("benchmark_ticker")
        base_currency = request.form.get("base_currency", "USD")

        if not start_date or not benchmark_ticker:
            return jsonify({"error": "시작 일자와 벤치마크 티커를 입력해주세요."}), 400

        # CSV 파일 파싱
        csv_content = file.read().decode("utf-8")
        print(f"📄 CSV content length: {len(csv_content)} bytes")

        portfolio_df = pd.read_csv(io.StringIO(csv_content))

        # 필수 컬럼 확인
        required_columns = ["티커", "보유량", "국가", "분류"]
        missing_columns = [
            col for col in required_columns if col not in portfolio_df.columns
        ]

        if missing_columns:
            return (
                jsonify(
                    {
                        "error": f'CSV 파일에 다음 컬럼이 필요합니다: {", ".join(missing_columns)}'
                    }
                ),
                400,
            )

        # 동일 티커 병합
        portfolio_df = merge_duplicate_tickers(portfolio_df)

        # 날짜 변환
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")

        # 포트폴리오 수익률 계산 (기준 통화 전달)
        print(f"📊 Starting portfolio calculation with {len(portfolio_df)} items...")
        result = calculate_portfolio_returns(
            portfolio_df, start_date_obj, base_currency
        )

        if result is None or result[0] is None:
            print("❌ Portfolio calculation failed - no valid data")
            
            # 실패한 티커 정보 추출
            failed_tickers = result[5] if result and len(result) > 5 else []
            
            error_msg = "포트폴리오 데이터를 가져올 수 없습니다."
            if failed_tickers:
                error_msg += f" 다음 티커에 문제가 있습니다: {', '.join(failed_tickers)}"
            else:
                error_msg += " 가능한 원인: 1) 모든 티커가 유효하지 않음, 2) 시작 날짜가 너무 최근, 3) 네트워크 오류"
            
            return jsonify({"error": error_msg}), 400

        (
            portfolio_returns,
            portfolio_series,
            portfolio_data,
            cash_holdings,
            total_initial_value_with_cash,
            failed_tickers,  # 실패한 티커 리스트 받기
        ) = result

        print(
            f"✓ Portfolio calculation successful: {len(portfolio_data)} stocks, {len(cash_holdings)} cash items"
        )
        
        # 일부 티커가 실패한 경우 경고 메시지 추가
        warning_msg = None
        if failed_tickers:
            warning_msg = f"⚠️ 다음 티커의 데이터를 가져올 수 없어 제외되었습니다: {', '.join(failed_tickers)}"
            print(f"⚠️ Warning: Some tickers failed: {failed_tickers}")

        # 벤치마크 데이터 가져오기
        print(f"📊 Fetching benchmark data: {benchmark_ticker}")
        benchmark_data = fetch_stock_data(
            benchmark_ticker, start_date_obj, datetime.now()
        )

        if benchmark_data is None:
            print(f"❌ Failed to fetch benchmark data for {benchmark_ticker}")
            return (
                jsonify(
                    {
                        "error": f'벤치마크 티커 "{benchmark_ticker}"의 데이터를 가져올 수 없습니다. 티커 이름을 확인하세요.'
                    }
                ),
                400,
            )

        # 벤치마크 데이터도 start_date 이후만 사용
        print(f"📊 Benchmark data before filter: {len(benchmark_data)} days")
        
        # fetch_stock_data가 timezone 없는 데이터 반환하므로 단순 비교
        start_date_for_filter = pd.to_datetime(start_date_obj)
        
        benchmark_data = benchmark_data[benchmark_data.index >= start_date_for_filter]
        print(f"📊 Benchmark data after filter: {len(benchmark_data)} days")
        if len(benchmark_data) > 0:
            print(f"    Date range: {benchmark_data.index[0].date()} to {benchmark_data.index[-1].date()}")
        else:
            print(f"    ❌ No benchmark data after filtering!")
            return (
                jsonify(
                    {"error": f"벤치마크 데이터가 시작 날짜({start_date}) 이후에 없습니다. 날짜를 확인하세요."}
                ),
                400,
            )

        benchmark_returns = benchmark_data.pct_change().dropna()
        
        print(f"📊 Returns comparison:")
        print(f"  Portfolio returns: {len(portfolio_returns)} days")
        print(f"  Benchmark returns: {len(benchmark_returns)} days")
        
        if len(benchmark_returns) == 0:
            print(f"    ❌ No benchmark returns after pct_change!")
            return (
                jsonify(
                    {"error": "벤치마크 수익률을 계산할 수 없습니다. 데이터가 충분하지 않습니다."}
                ),
                400,
            )

        # 지표 계산
        metrics = calculate_metrics(portfolio_returns, benchmark_returns)

        if metrics is None:
            print(f"    ❌ calculate_metrics returned None")
            return (
                jsonify(
                    {"error": "지표를 계산할 수 없습니다. 포트폴리오와 벤치마크의 날짜 범위가 겹치지 않습니다."}
                ),
                400,
            )

        # 차트 데이터 준비
        chart_data = prepare_chart_data(
            portfolio_returns, benchmark_returns, portfolio_series
        )

        # 현재 환율 정보 (보유 종목 테이블과 요약에 사용)
        exchange_rate = get_current_exchange_rate()

        # 도넛 차트 데이터 준비
        allocation_data = prepare_allocation_data(
            portfolio_data, cash_holdings, total_initial_value_with_cash
        )

        # 보유 종목 테이블 데이터 준비
        holdings_table = prepare_holdings_table(
            portfolio_data, cash_holdings, base_currency, exchange_rate
        )

        # 포트폴리오 요약 정보
        current_value = portfolio_series.iloc[-1]
        initial_value = portfolio_series.iloc[0]

        # 현금 총액 계산
        total_cash = sum(cash["value"] for cash in cash_holdings.values())

        # 현금 포함한 현재 총 가치
        current_value_with_cash = current_value + total_cash
        initial_value_with_cash = total_initial_value_with_cash

        summary = {
            "initial_value": round(initial_value, 2),
            "current_value": round(current_value, 2),
            "initial_value_with_cash": round(initial_value_with_cash, 2),
            "current_value_with_cash": round(current_value_with_cash, 2),
            "total_cash": round(total_cash, 2),
            "total_return": round((current_value / initial_value - 1) * 100, 2),
            "total_return_with_cash": round(
                (current_value_with_cash / initial_value_with_cash - 1) * 100, 2
            ),
            "start_date": start_date,
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "benchmark": benchmark_ticker,
            "num_holdings": len(portfolio_df),
            "base_currency": base_currency,
            "exchange_rate": round(exchange_rate, 2),
        }

        return jsonify(
            {
                "metrics": metrics,
                "chart_data": chart_data,
                "summary": summary,
                "allocation_data": allocation_data,
                "holdings_table": holdings_table,
                "warning": warning_msg,  # 경고 메시지 추가
            }
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        print("=" * 80)
        print("❌ ERROR in /analyze endpoint:")
        print(error_trace)
        print("=" * 80)
        return (
            jsonify(
                {
                    "error": f"분석 중 오류가 발생했습니다: {str(e)}\n\n서버 터미널에서 상세 로그를 확인하세요."
                }
            ),
            500,
        )


@app.route("/api/ai-analysis", methods=["POST"])
def ai_analysis():
    """OpenAI를 사용한 포트폴리오 AI 분석"""
    try:
        import json
        from datetime import datetime, timedelta
        
        # 클라이언트 IP 가져오기
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ',' in client_ip:
            client_ip = client_ip.split(',')[0].strip()
        
        current_time = datetime.now()
        
        # Rate limiting 체크 (3분에 1번)
        if client_ip in ai_analysis_rate_limit:
            last_request = ai_analysis_rate_limit[client_ip]
            time_diff = (current_time - last_request).total_seconds()
            
            if time_diff < 180:
                remaining_seconds = int(180 - time_diff)
                return jsonify({
                    "success": False,
                    "error": f"AI 분석은 3분에 1번만 가능합니다. {remaining_seconds}초 후에 다시 시도해주세요.",
                    "rate_limited": True,
                    "remaining_seconds": remaining_seconds
                }), 429
        
        data = request.json
        
        # 필요한 데이터 추출
        holdings = data.get("holdings", [])
        metrics = data.get("metrics", {})
        summary = data.get("summary", {})
        benchmark = summary.get("benchmark", "Unknown")
        
        # 캐시 키 생성 (holdings의 티커와 비중으로)
        cache_key = json.dumps({
            "holdings": sorted([(h['ticker'], h['weight']) for h in holdings]),
            "cagr": metrics.get('cagr'),
            "sharpe": metrics.get('sharpe_ratio'),
            "benchmark": benchmark
        }, sort_keys=True)
        
        # 캐시에서 확인 (같은 포트폴리오는 재분석하지 않음)
        if client_ip in ai_analysis_cache:
            cached_data = ai_analysis_cache[client_ip]
            if cached_data.get("cache_key") == cache_key:
                print(f"✅ Returning cached AI analysis for IP: {client_ip}")
                return jsonify({
                    "success": True,
                    "analysis": cached_data["result"],
                    "cached": True
                })
        
        # 벤치마크 이름 매핑
        benchmark_names = {
            "SPY": "S&P 500",
            "^GSPC": "S&P 500",
            "QQQ": "NASDAQ 100",
            "^NDX": "NASDAQ 100",
            "069500.KS": "KOSPI 200"
        }
        benchmark_name = benchmark_names.get(benchmark, benchmark)
        
        # 보유 종목 정보 포맷팅
        holdings_text = "\n".join([
            f"- {h['ticker']} ({h['name']}): {h['weight']}%"
            for h in holdings
        ])
        
        # 프롬프트 구성
        prompt = f"""당신은 전문 포트폴리오 분석가입니다. 다음 포트폴리오를 분석해주세요.

## 보유 종목 및 비중
{holdings_text}

## 성과 지표
- 연평균 수익률 (CAGR): {metrics.get('cagr', 'N/A')}%
- 샤프 비율: {metrics.get('sharpe_ratio', 'N/A')}
- 소티노 비율: {metrics.get('sortino_ratio', 'N/A')}
- 알파: {metrics.get('alpha', 'N/A')}%
- 베타: {metrics.get('beta', 'N/A')}
- 변동성: {metrics.get('volatility', 'N/A')}%
- 최대 낙폭 (MDD): {metrics.get('max_drawdown', 'N/A')}%

## 벤치마크 ({benchmark_name}) 대비
- 벤치마크 연평균 수익률: {metrics.get('benchmark_annual_return', 'N/A')}%
- 벤치마크 샤프 비율: {metrics.get('benchmark_sharpe_ratio', 'N/A')}
- 벤치마크 소티노 비율: {metrics.get('benchmark_sortino_ratio', 'N/A')}
- 벤치마크 변동성: {metrics.get('benchmark_volatility', 'N/A')}%

다음 항목들을 **마크다운 형식**으로 분석해주세요:

1. **포트폴리오 강점 분석** (## 제목 사용)
   - 벤치마크 대비 우수한 점
   - 위험 대비 수익률이 좋은 이유
   - 잘 구성된 부분

2. **포트폴리오 약점 및 위험 요소** (## 제목 사용)
   - 개선이 필요한 지표
   - 잠재적 위험 요소
   - 벤치마크 대비 부족한 점

3. **개선 제안** (## 제목 사용)
   - 구체적인 개선 방안
   - 리밸런싱 제안
   - 추가/제거 고려 종목

답변은 반드시 마크다운 형식으로 작성하고, 이모지를 적절히 활용하여 가독성을 높여주세요.
전문적이면서도 이해하기 쉽게 작성해주세요."""

        # OpenAI API 호출
        print("🤖 Calling OpenAI API...")
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 20년 경력의 전문 포트폴리오 분석가입니다. 데이터 기반으로 객관적이고 실용적인 조언을 제공합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        analysis_result = response.choices[0].message.content
        
        print("✅ OpenAI API call successful")
        
        # Rate limit 업데이트
        ai_analysis_rate_limit[client_ip] = current_time
        
        # 캐시 저장
        ai_analysis_cache[client_ip] = {
            "cache_key": cache_key,
            "result": analysis_result,
            "timestamp": current_time
        }
        
        return jsonify({
            "success": True,
            "analysis": analysis_result,
            "cached": False
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print("=" * 80)
        print("❌ ERROR in /api/ai-analysis endpoint:")
        print(error_trace)
        print("=" * 80)
        return jsonify({
            "success": False,
            "error": f"AI 분석 중 오류가 발생했습니다: {str(e)}"
        }), 500


if __name__ == "__main__":
    # 데이터베이스 초기화
    init_database()

    # Flask 앱 실행 (외부 접속 허용)
    app.run(host='0.0.0.0', debug=False, port=8000)
