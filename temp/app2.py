from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    Text,
    DateTime,
    func,
)
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqladmin import Admin, ModelView
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import io
import traceback
import sqlite3
import os
from dotenv import load_dotenv
import openai
from typing import Optional, Dict, Any
import json
from contextlib import asynccontextmanager

# .env 파일 로드
load_dotenv()


# Lifespan 이벤트 핸들러 (SQLAlchemy 2.0 호환)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 및 종료 시 실행되는 이벤트"""
    # Startup
    print("🚀 Starting Portfolio Analyzer FastAPI App...")
    init_database()
    print("✅ Database initialized")
    yield
    # Shutdown
    print("👋 Shutting down Portfolio Analyzer...")


# FastAPI 앱 생성
app = FastAPI(title="Portfolio Analyzer", version="2.0", lifespan=lifespan)

# Templates 설정
templates = Jinja2Templates(directory="templates")

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# AI 분석 rate limiting과 캐싱을 위한 딕셔너리
ai_analysis_cache: Dict[str, Dict[str, Any]] = {}
ai_analysis_rate_limit: Dict[str, datetime] = {}

# DB 파일 경로
DB_PATH = os.path.join(os.path.dirname(__file__), "stock_cache.db")

# SQLAlchemy 설정
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# SQLAlchemy Base 클래스 정의 (SQLAlchemy 2.0 스타일)
class Base(DeclarativeBase):
    pass


# SQLAlchemy 모델 정의
class StockPriceCache(Base):
    __tablename__ = "stock_price_cache"

    ticker = Column(String(20), primary_key=True)
    date = Column(String(10), primary_key=True)
    close_price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<StockPrice {self.ticker} {self.date}>"


class SavedPortfolio(Base):
    __tablename__ = "saved_portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    csv_content = Column(Text, nullable=False)
    start_date = Column(String(10))
    benchmark_ticker = Column(String(20))
    base_currency = Column(String(3))
    created_at = Column(DateTime, default=datetime.now)
    last_accessed = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Portfolio {self.name}>"


# SQLAdmin ModelView 정의
class StockPriceCacheAdmin(ModelView, model=StockPriceCache):
    column_list = [
        StockPriceCache.ticker,
        StockPriceCache.date,
        StockPriceCache.close_price,
        StockPriceCache.created_at,
    ]
    column_searchable_list = [StockPriceCache.ticker]
    column_sortable_list = [
        StockPriceCache.ticker,
        StockPriceCache.date,
        StockPriceCache.created_at,
    ]
    column_default_sort = [
        (StockPriceCache.ticker, False),
        (StockPriceCache.date, True),
    ]
    page_size = 50
    can_export = True


class SavedPortfolioAdmin(ModelView, model=SavedPortfolio):
    column_list = [
        SavedPortfolio.id,
        SavedPortfolio.name,
        SavedPortfolio.start_date,
        SavedPortfolio.benchmark_ticker,
        SavedPortfolio.base_currency,
        SavedPortfolio.created_at,
        SavedPortfolio.last_accessed,
    ]
    column_searchable_list = [SavedPortfolio.name, SavedPortfolio.benchmark_ticker]
    column_sortable_list = [
        SavedPortfolio.id,
        SavedPortfolio.name,
        SavedPortfolio.created_at,
        SavedPortfolio.last_accessed,
    ]
    column_default_sort = [(SavedPortfolio.last_accessed, True)]
    page_size = 50
    can_export = True
    # CSV 내용은 너무 길어서 리스트에서 제외
    column_details_exclude_list = []
    form_excluded_columns = []


# SQLAdmin 초기화
admin = Admin(app, engine)
admin.add_view(StockPriceCacheAdmin)
admin.add_view(SavedPortfolioAdmin)


# 데이터베이스 초기화
def init_database():
    """데이터베이스 초기화 및 테이블 생성"""
    Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========== 헬퍼 함수들 (app.py에서 가져옴) ==========


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
    except:
        return 1350.0


def fetch_stock_data(ticker, start_date, end_date):
    """Yahoo Finance에서 주가 데이터 가져오기 (DB 캐시 활용)

    상장일이 start_date보다 늦은 경우에도 상장 이후 데이터를 반환하여
    fill_missing_dates에서 상장일 이전 구간을 상장 시 가격으로 채울 수 있도록 함
    """
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
            # 또는 상장일이 start_date 이후인 경우에도 데이터 반환
            if cached_start <= expected_start + timedelta(
                days=7
            ) and cached_end >= expected_end - timedelta(days=7):
                return cached_data
            elif cached_end >= expected_end - timedelta(days=7):
                # 상장일이 늦어도 end_date까지 데이터가 있으면 반환
                return cached_data

        # 3. 캐시에 없으면 Yahoo Finance에서 가져오기
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        # 데이터가 비어있는 경우 (상장일이 늦을 수 있음)
        if data.empty:
            # 더 넓은 범위로 다시 시도 (최근 5년 또는 상장일부터)
            print(
                f"  ⚠ No data for {ticker} in requested range, trying broader range..."
            )
            data = stock.history(period="5y")

            if data.empty:
                print(f"  ❌ {ticker}: Still no data available")
                return None
            else:
                # 상장일 이후 데이터가 있음
                listing_date = data.index.min()
                print(
                    f"  ℹ {ticker} listing date appears to be around {listing_date.date()}"
                )

        price_data = data["Close"]

        # timezone 제거 (캐시된 데이터와 일관성 유지)
        if hasattr(price_data.index, "tz") and price_data.index.tz is not None:
            price_data.index = price_data.index.tz_localize(None)

        # NaN 값 제거
        price_data = price_data.dropna()

        if len(price_data) == 0:
            print(f"  ❌ {ticker}: No valid price data after cleaning")
            return None

        # 4. 새로 가져온 데이터를 DB에 저장
        save_prices_to_cache(ticker, price_data)

        return price_data

    except Exception as e:
        print(f"  ❌ Error fetching {ticker}: {e}")
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
        print(
            f"  🔍 Index type: {type(price_series.index)}, dtype: {price_series.index.dtype}"
        )

        # 원본 데이터의 인덱스를 timezone-naive로 변환
        if hasattr(price_series.index, "tz") and price_series.index.tz is not None:
            print(f"  🌐 Converting from timezone: {price_series.index.tz}")
            price_series.index = price_series.index.tz_localize(None)

        # 인덱스가 DatetimeIndex인지 확인
        if not isinstance(price_series.index, pd.DatetimeIndex):
            print(f"  🔄 Converting index to DatetimeIndex")
            price_series.index = pd.to_datetime(price_series.index)

        # 전체 날짜 범위 생성 (모든 날짜 포함)
        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

        print(f"  📆 Expanded range: {len(all_dates)} days")
        print(
            f"  📊 First original date: {price_series.index[0]}, Last: {price_series.index[-1]}"
        )

        # 기존 데이터를 전체 날짜 범위로 확장
        price_series_filled = price_series.reindex(all_dates)

        # 데이터가 있는 첫 날짜 확인 (상장일)
        first_valid_date = price_series_filled.first_valid_index()

        if first_valid_date is None:
            print("  ❌ No valid dates found after reindex")
            print(
                f"  🔍 Checking overlap: orig min={price_series.index.min()}, max={price_series.index.max()}"
            )
            print(
                f"  🔍 Checking overlap: new min={all_dates.min()}, max={all_dates.max()}"
            )
            # 그냥 원본 데이터 반환 (날짜 확장 없이)
            return price_series

        # 상장 이전 데이터: 상장 후 첫 가격으로 채움
        first_price = price_series_filled[first_valid_date]
        price_series_filled.loc[:first_valid_date] = first_price

        # 상장 이후 빈 데이터: 선형 보간
        price_series_filled = price_series_filled.interpolate(
            method="linear", limit_direction="forward"
        )

        # 아직도 빈 값이 있다면 (끝 부분) forward fill
        price_series_filled = price_series_filled.ffill()

        # 그래도 남아있는 NaN은 backward fill
        price_series_filled = price_series_filled.bfill()

        filled_count = len(all_dates) - len(price_series)
        print(
            f"  ✓ Filled {filled_count} missing dates (total: {len(price_series_filled)} days)"
        )

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

        # fill_missing_dates를 호출하여 상장일 이전 데이터를 상장 시 가격으로 채움
        print(f"  🔄 Filling missing dates for {ticker}...")
        price_data = fill_missing_dates(price_data, start_date, end_date)

        if price_data is None or len(price_data) == 0:
            print(f"⚠ Skipping {ticker}: Failed to process price data")
            failed_tickers.append(ticker)
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
        return (
            None,
            None,
            None,
            cash_holdings,
            total_initial_value_with_cash,
            failed_tickers,
        )

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
    print(
        f"  Actual start date: {all_dates[0].date() if len(all_dates) > 0 else 'N/A'}"
    )
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
    print(
        f"  Total return: {(portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1) * 100:.2f}%"
    )

    # 일일 수익률 계산
    returns = portfolio_series.pct_change().dropna()

    return (
        returns,
        portfolio_data,
        portfolio_series,
        cash_holdings,
        total_initial_value_with_cash,
        failed_tickers,
    )


def calculate_weighted_annual_return(portfolio_returns):
    """연환산 수익률 계산 (복리)"""
    try:
        if len(portfolio_returns) == 0:
            return 0.0

        # 누적 수익률 계산
        cumulative_return = (1 + portfolio_returns).prod() - 1

        # 거래일 수
        trading_days = len(portfolio_returns)

        # 연환산 (252 거래일 기준)
        annual_return = (1 + cumulative_return) ** (252 / trading_days) - 1

        return annual_return * 100  # 퍼센트로 반환

    except Exception as e:
        print(f"Error calculating annual return: {e}")
        return 0.0


def calculate_metrics(portfolio_returns, benchmark_returns):
    """포트폴리오 성과 지표 계산"""

    metrics = {}

    try:
        # 1. 연환산 수익률
        metrics["annual_return"] = calculate_weighted_annual_return(portfolio_returns)
        metrics["benchmark_annual_return"] = calculate_weighted_annual_return(
            benchmark_returns
        )

        # 2. 누적 수익률
        portfolio_cumulative = (1 + portfolio_returns).prod() - 1
        benchmark_cumulative = (1 + benchmark_returns).prod() - 1
        metrics["cumulative_return"] = portfolio_cumulative * 100
        metrics["benchmark_cumulative_return"] = benchmark_cumulative * 100

        # 3. 변동성 (연환산)
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        metrics["volatility"] = portfolio_volatility * 100
        metrics["benchmark_volatility"] = benchmark_volatility * 100

        # 4. 샤프 비율 (무위험 수익률 0% 가정)
        if portfolio_volatility > 0:
            sharpe_ratio = (metrics["annual_return"] / 100) / portfolio_volatility
            metrics["sharpe_ratio"] = sharpe_ratio
        else:
            metrics["sharpe_ratio"] = 0

        if benchmark_volatility > 0:
            benchmark_sharpe = (
                metrics["benchmark_annual_return"] / 100
            ) / benchmark_volatility
            metrics["benchmark_sharpe_ratio"] = benchmark_sharpe
        else:
            metrics["benchmark_sharpe_ratio"] = 0

        # 5. 최대 낙폭 (Maximum Drawdown)
        portfolio_cum_returns = (1 + portfolio_returns).cumprod()
        portfolio_running_max = portfolio_cum_returns.expanding().max()
        portfolio_drawdown = (
            portfolio_cum_returns - portfolio_running_max
        ) / portfolio_running_max
        metrics["max_drawdown"] = portfolio_drawdown.min() * 100

        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        benchmark_running_max = benchmark_cum_returns.expanding().max()
        benchmark_drawdown = (
            benchmark_cum_returns - benchmark_running_max
        ) / benchmark_running_max
        metrics["benchmark_max_drawdown"] = benchmark_drawdown.min() * 100

        # 6. 베타 (Beta)
        # 공분산 / 벤치마크 분산
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
            metrics["beta"] = beta
        else:
            metrics["beta"] = 0

        # 7. 알파 (Alpha) - 연환산
        # 알파 = 포트폴리오 수익률 - (무위험 수익률 + 베타 * (벤치마크 수익률 - 무위험 수익률))
        # 무위험 수익률 = 0으로 가정
        if "beta" in metrics:
            alpha = metrics["annual_return"] - (
                metrics["beta"] * metrics["benchmark_annual_return"]
            )
            metrics["alpha"] = alpha
        else:
            metrics["alpha"] = 0

        # 8. 정보 비율 (Information Ratio)
        # (포트폴리오 수익률 - 벤치마크 수익률) / 추적오차
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error > 0:
            information_ratio = (
                metrics["annual_return"] - metrics["benchmark_annual_return"]
            ) / (tracking_error * 100)
            metrics["information_ratio"] = information_ratio
        else:
            metrics["information_ratio"] = 0

        # 9. 승률 (Win Rate) - 벤치마크 대비
        outperformance_days = (portfolio_returns > benchmark_returns).sum()
        total_days = len(portfolio_returns)
        metrics["win_rate"] = (
            (outperformance_days / total_days * 100) if total_days > 0 else 0
        )

        # 10. 소티노 비율 (Sortino Ratio)
        # 샤프 비율과 유사하지만 하방 변동성만 고려
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std > 0:
                sortino_ratio = (metrics["annual_return"] / 100) / downside_std
                metrics["sortino_ratio"] = sortino_ratio
            else:
                metrics["sortino_ratio"] = 0
        else:
            metrics["sortino_ratio"] = 0

        # 11. 칼마 비율 (Calmar Ratio)
        # 연환산 수익률 / 절대값(최대 낙폭)
        if metrics["max_drawdown"] != 0:
            calmar_ratio = (metrics["annual_return"] / 100) / abs(
                metrics["max_drawdown"] / 100
            )
            metrics["calmar_ratio"] = calmar_ratio
        else:
            metrics["calmar_ratio"] = 0

        # 모든 값을 Python float로 변환 (numpy 타입 제거)
        metrics = {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in metrics.items()
        }

        return metrics

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        traceback.print_exc()
        return {}


def prepare_chart_data(portfolio_returns, benchmark_returns, portfolio_series):
    """차트 데이터 준비"""
    try:
        # 누적 수익률 계산
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()

        # 날짜를 문자열로 변환
        dates = [date.strftime("%Y-%m-%d") for date in portfolio_cumulative.index]

        chart_data = {
            "dates": dates,
            "portfolio": [
                float(val - 1) * 100 for val in portfolio_cumulative.values
            ],  # 퍼센트로 변환
            "benchmark": [float(val - 1) * 100 for val in benchmark_cumulative.values],
        }

        return chart_data

    except Exception as e:
        print(f"Error preparing chart data: {e}")
        traceback.print_exc()
        return None


def prepare_allocation_data(
    portfolio_data, cash_holdings, base_currency, exchange_rate
):
    """자산 배분 데이터 준비 (국가별, 자산별)"""
    try:
        country_allocation = {}
        asset_allocation = {}

        # 주식 자산
        for ticker, data in portfolio_data.items():
            country = data["country"]
            asset_class = data.get("asset_class", "주식")
            current_value = data["prices"].iloc[-1] * data["quantity"]

            # 국가별 집계
            if country in country_allocation:
                country_allocation[country] += current_value
            else:
                country_allocation[country] = current_value

            # 자산별 집계
            if asset_class in asset_allocation:
                asset_allocation[asset_class] += current_value
            else:
                asset_allocation[asset_class] = current_value

        # 현금 자산 추가
        for ticker, cash_data in cash_holdings.items():
            country = cash_data["country"]
            cash_value = cash_data["value"]

            # 국가별 집계
            if country in country_allocation:
                country_allocation[country] += cash_value
            else:
                country_allocation[country] = cash_value

            # 자산별 집계
            if "현금" in asset_allocation:
                asset_allocation["현금"] += cash_value
            else:
                asset_allocation["현금"] = cash_value

        # 상위 N개만 표시하는 헬퍼 함수
        def get_top_allocations(allocation_dict, top_n=10):
            sorted_items = sorted(
                allocation_dict.items(), key=lambda x: x[1], reverse=True
            )
            if len(sorted_items) <= top_n:
                # float로 변환하여 반환
                return {k: float(v) for k, v in sorted_items}
            else:
                top_items = {k: float(v) for k, v in sorted_items[:top_n]}
                others_value = sum(value for _, value in sorted_items[top_n:])
                if others_value > 0:
                    top_items["기타"] = float(others_value)
                return top_items

        # 상위 자산만 필터링
        country_allocation = get_top_allocations(country_allocation, top_n=10)
        asset_allocation = get_top_allocations(asset_allocation, top_n=10)

        return {
            "country": country_allocation,
            "asset": asset_allocation,
        }

    except Exception as e:
        print(f"Error preparing allocation data: {e}")
        traceback.print_exc()
        return {"country": {}, "asset": {}}


def prepare_holdings_table(portfolio_data, cash_holdings, base_currency, exchange_rate):
    """보유 종목 테이블 데이터 준비"""
    try:
        holdings = []

        # 주식 자산
        for ticker, data in portfolio_data.items():
            initial_price = float(data["prices"].iloc[0])
            current_price = float(data["prices"].iloc[-1])
            quantity = float(data["quantity"])

            initial_value = initial_price * quantity
            current_value = current_price * quantity
            gain_loss = current_value - initial_value
            gain_loss_pct = (
                (gain_loss / initial_value * 100) if initial_value > 0 else 0
            )

            holdings.append(
                {
                    "ticker": str(ticker),
                    "name": str(data.get("name", ticker)),
                    "quantity": float(quantity),
                    "initial_price": float(initial_price),
                    "current_price": float(current_price),
                    "initial_value": float(initial_value),
                    "current_value": float(current_value),
                    "gain_loss": float(gain_loss),
                    "gain_loss_pct": float(gain_loss_pct),
                    "country": str(data["country"]),
                    "asset_class": str(data.get("asset_class", "주식")),
                }
            )

        # 현금 자산 추가
        for ticker, cash_data in cash_holdings.items():
            holdings.append(
                {
                    "ticker": str(ticker),
                    "name": "현금",
                    "quantity": 1.0,
                    "initial_price": float(cash_data["value"]),
                    "current_price": float(cash_data["value"]),
                    "initial_value": float(cash_data["value"]),
                    "current_value": float(cash_data["value"]),
                    "gain_loss": 0.0,
                    "gain_loss_pct": 0.0,
                    "country": str(cash_data["country"]),
                    "asset_class": "현금",
                }
            )

        # 현재 가치 기준으로 정렬
        holdings.sort(key=lambda x: x["current_value"], reverse=True)

        return holdings

    except Exception as e:
        print(f"Error preparing holdings table: {e}")
        traceback.print_exc()
        return []


# ========== FastAPI 라우트 ==========


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/cache-stats")
async def get_cache_stats(db: Session = Depends(get_db)):
    """캐시 통계 API"""
    try:
        # 티커별 캐시 데이터 개수 조회
        result = (
            db.query(
                StockPriceCache.ticker,
                func.count(StockPriceCache.date).label("count"),
                func.min(StockPriceCache.date).label("min_date"),
                func.max(StockPriceCache.date).label("max_date"),
            )
            .group_by(StockPriceCache.ticker)
            .all()
        )

        cache_stats = []
        for row in result:
            cache_stats.append(
                {
                    "ticker": row.ticker,
                    "count": row.count,
                    "min_date": row.min_date,
                    "max_date": row.max_date,
                }
            )

        return JSONResponse(
            content={
                "success": True,
                "total_tickers": len(cache_stats),
                "cache_stats": cache_stats,
            }
        )

    except Exception as e:
        print(f"Error in get_cache_stats: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@app.post("/api/save-portfolio")
async def save_portfolio(
    name: str = Form(...),
    csv_content: str = Form(...),
    start_date: str = Form(...),
    benchmark_ticker: str = Form(...),
    base_currency: str = Form(...),
    db: Session = Depends(get_db),
):
    """포트폴리오 저장 API"""
    try:
        # 새 포트폴리오 생성
        portfolio = SavedPortfolio(
            name=name,
            csv_content=csv_content,
            start_date=start_date,
            benchmark_ticker=benchmark_ticker,
            base_currency=base_currency,
        )

        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)

        return JSONResponse(
            content={
                "success": True,
                "portfolio_id": portfolio.id,
                "message": "포트폴리오가 저장되었습니다.",
            }
        )

    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@app.get("/ranking", response_class=HTMLResponse)
async def ranking_page(request: Request):
    """랭킹 페이지"""
    return templates.TemplateResponse("ranking.html", {"request": request})


@app.get("/api/rankings")
async def get_rankings(db: Session = Depends(get_db)):
    """포트폴리오 랭킹 조회 API"""
    try:
        portfolios = db.query(SavedPortfolio).all()

        if not portfolios:
            return JSONResponse(content={"success": True, "rankings": []})

        # 각 포트폴리오 분석
        rankings = []
        for portfolio in portfolios:
            try:
                # CSV 파싱
                csv_data = io.StringIO(portfolio.csv_content)
                portfolio_df = pd.read_csv(csv_data)

                # 중복 티커 병합
                portfolio_df = merge_duplicate_tickers(portfolio_df)

                # 시작 날짜 파싱
                start_date_obj = datetime.strptime(portfolio.start_date, "%Y-%m-%d")

                # 포트폴리오 수익률 계산
                (
                    portfolio_returns,
                    portfolio_data,
                    portfolio_series,
                    cash_holdings,
                    total_initial_value_with_cash,
                    failed_tickers,
                ) = calculate_portfolio_returns(
                    portfolio_df, start_date_obj, portfolio.base_currency
                )

                if portfolio_returns is None:
                    continue

                # 벤치마크 데이터
                benchmark_data = fetch_stock_data(
                    portfolio.benchmark_ticker, start_date_obj, datetime.now()
                )

                if benchmark_data is None:
                    continue

                # 벤치마크 데이터도 fill_missing_dates 호출
                benchmark_data = fill_missing_dates(
                    benchmark_data, start_date_obj, datetime.now()
                )

                if benchmark_data is None or len(benchmark_data) == 0:
                    continue

                start_date_for_filter = pd.to_datetime(start_date_obj)
                benchmark_data = benchmark_data[
                    benchmark_data.index >= start_date_for_filter
                ]
                benchmark_returns = benchmark_data.pct_change().dropna()

                # 지표 계산
                metrics = calculate_metrics(portfolio_returns, benchmark_returns)

                rankings.append(
                    {
                        "id": portfolio.id,
                        "name": portfolio.name,
                        "annual_return": metrics.get("annual_return", 0),
                        "cumulative_return": metrics.get("cumulative_return", 0),
                        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                        "max_drawdown": metrics.get("max_drawdown", 0),
                        "volatility": metrics.get("volatility", 0),
                        "alpha": metrics.get("alpha", 0),
                        "beta": metrics.get("beta", 0),
                        "start_date": portfolio.start_date,
                        "benchmark_ticker": portfolio.benchmark_ticker,
                        "created_at": portfolio.created_at.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                )

            except Exception as e:
                print(f"Error processing portfolio {portfolio.id}: {e}")
                continue

        # 연환산 수익률 기준 정렬
        rankings.sort(key=lambda x: x["annual_return"], reverse=True)

        return JSONResponse(content={"success": True, "rankings": rankings})

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@app.get("/portfolio/{portfolio_id}", response_class=HTMLResponse)
async def portfolio_view(
    request: Request, portfolio_id: int, db: Session = Depends(get_db)
):
    """저장된 포트폴리오 조회"""
    try:
        portfolio = (
            db.query(SavedPortfolio).filter(SavedPortfolio.id == portfolio_id).first()
        )

        if not portfolio:
            raise HTTPException(
                status_code=404, detail="포트폴리오를 찾을 수 없습니다."
            )

        # 마지막 접속 시간 업데이트
        portfolio.last_accessed = datetime.now()
        db.commit()

        return templates.TemplateResponse(
            "portfolio_view.html",
            {
                "request": request,
                "portfolio_id": portfolio_id,
                "portfolio_name": portfolio.name,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    start_date: str = Form(...),
    benchmark_ticker: str = Form(...),
    base_currency: str = Form("USD"),
):
    """포트폴리오 분석 API"""
    try:
        print("=" * 80)
        print("📊 Starting portfolio analysis...")
        print(f"  Start date: {start_date}")
        print(f"  Benchmark: {benchmark_ticker}")
        print(f"  Base currency: {base_currency}")
        print("=" * 80)

        # CSV 파일 읽기
        contents = await file.read()
        csv_data = io.StringIO(contents.decode("utf-8"))
        portfolio_df = pd.read_csv(csv_data)

        print(f"\n📋 Portfolio data loaded:")
        print(f"  Rows: {len(portfolio_df)}")
        print(f"  Columns: {list(portfolio_df.columns)}")

        # 중복 티커 병합
        portfolio_df = merge_duplicate_tickers(portfolio_df)
        print(f"  After merging duplicates: {len(portfolio_df)} unique tickers")

        # 시작 날짜 파싱
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요."
                },
            )

        # 포트폴리오 수익률 계산
        print(f"\n💼 Calculating portfolio returns...")
        (
            portfolio_returns,
            portfolio_data,
            portfolio_series,
            cash_holdings,
            total_initial_value_with_cash,
            failed_tickers,
        ) = calculate_portfolio_returns(portfolio_df, start_date_obj, base_currency)

        if portfolio_returns is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "포트폴리오 데이터를 처리할 수 없습니다. CSV 파일을 확인하세요."
                },
            )

        warning_msg = ""
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
            return JSONResponse(
                status_code=400,
                content={
                    "error": f'벤치마크 티커 "{benchmark_ticker}"의 데이터를 가져올 수 없습니다. 티커 이름을 확인하세요.'
                },
            )

        # 벤치마크 데이터도 fill_missing_dates 호출
        print(f"  🔄 Filling missing dates for benchmark {benchmark_ticker}...")
        benchmark_data = fill_missing_dates(
            benchmark_data, start_date_obj, datetime.now()
        )

        if benchmark_data is None or len(benchmark_data) == 0:
            print(f"❌ Failed to process benchmark data for {benchmark_ticker}")
            return JSONResponse(
                status_code=400,
                content={"error": f"벤치마크 데이터 처리 중 오류가 발생했습니다."},
            )

        # 벤치마크 데이터도 start_date 이후만 사용
        print(f"📊 Benchmark data before filter: {len(benchmark_data)} days")

        # fetch_stock_data가 timezone 없는 데이터 반환하므로 단순 비교
        start_date_for_filter = pd.to_datetime(start_date_obj)

        benchmark_data = benchmark_data[benchmark_data.index >= start_date_for_filter]
        print(f"📊 Benchmark data after filter: {len(benchmark_data)} days")
        if len(benchmark_data) > 0:
            print(
                f"    Date range: {benchmark_data.index[0].date()} to {benchmark_data.index[-1].date()}"
            )
        else:
            print(f"    ❌ No benchmark data after filtering!")
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"벤치마크 데이터가 시작 날짜({start_date}) 이후에 없습니다. 날짜를 확인하세요."
                },
            )

        benchmark_returns = benchmark_data.pct_change().dropna()

        print(f"📊 Returns comparison:")
        print(f"  Portfolio returns: {len(portfolio_returns)} days")
        print(f"  Benchmark returns: {len(benchmark_returns)} days")

        if len(benchmark_returns) == 0:
            print(f"    ❌ No benchmark returns after pct_change!")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "벤치마크 수익률을 계산할 수 없습니다. 데이터가 충분하지 않습니다."
                },
            )

        # 날짜 범위 맞추기
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        print(f"  Common dates: {len(common_dates)} days")

        if len(common_dates) < 20:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "포트폴리오와 벤치마크의 공통 거래일이 충분하지 않습니다. (최소 20일 필요)"
                },
            )

        portfolio_returns = portfolio_returns[common_dates]
        benchmark_returns = benchmark_returns[common_dates]

        # 성과 지표 계산
        print(f"\n📈 Calculating metrics...")
        metrics = calculate_metrics(portfolio_returns, benchmark_returns)

        # 차트 데이터 준비
        print(f"📊 Preparing chart data...")
        chart_data = prepare_chart_data(
            portfolio_returns, benchmark_returns, portfolio_series
        )

        # 자산 배분 데이터
        print(f"🥧 Preparing allocation data...")
        exchange_rate = get_current_exchange_rate()
        allocation_data = prepare_allocation_data(
            portfolio_data, cash_holdings, base_currency, exchange_rate
        )

        # 보유 종목 테이블
        print(f"📋 Preparing holdings table...")
        holdings = prepare_holdings_table(
            portfolio_data, cash_holdings, base_currency, exchange_rate
        )

        print(f"\n✅ Analysis complete!")
        print("=" * 80)

        # 결과 반환
        return JSONResponse(
            content={
                "success": True,
                "metrics": metrics,
                "chart_data": chart_data,
                "allocation": allocation_data,
                "holdings": holdings,
                "warning": warning_msg,
            }
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        print("=" * 80)
        print("❌ ERROR in /analyze endpoint:")
        print(error_trace)
        print("=" * 80)
        return JSONResponse(
            status_code=500, content={"error": f"분석 중 오류가 발생했습니다: {str(e)}"}
        )


@app.post("/api/ai-analysis")
async def ai_analysis(request: Request):
    """AI 분석 API (OpenAI)"""
    try:
        # 요청 데이터 읽기
        body = await request.json()
        metrics = body.get("metrics", {})
        holdings = body.get("holdings", [])

        # Rate limiting 체크 (IP 기반)
        client_ip = request.client.host
        current_time = datetime.now()

        # 1분에 1회로 제한
        if client_ip in ai_analysis_rate_limit:
            last_request = ai_analysis_rate_limit[client_ip]
            if (current_time - last_request).total_seconds() < 60:
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "error": "요청이 너무 빠릅니다. 1분 후 다시 시도하세요.",
                    },
                )

        # 캐시 체크 (IP + 메트릭 해시)
        cache_key = f"{client_ip}_{json.dumps(metrics, sort_keys=True)}"
        if client_ip in ai_analysis_cache:
            cached = ai_analysis_cache[client_ip]
            if cached["cache_key"] == cache_key:
                # 캐시 유효 시간: 1시간
                if (current_time - cached["timestamp"]).total_seconds() < 3600:
                    print(f"✅ Using cached AI analysis for IP: {client_ip}")
                    return JSONResponse(
                        content={
                            "success": True,
                            "analysis": cached["result"],
                            "cached": True,
                        }
                    )

        # OpenAI API 호출
        print(f"📡 Calling OpenAI API for IP: {client_ip}")

        # 프롬프트 구성
        prompt = f"""
다음은 투자 포트폴리오의 성과 분석 결과입니다:

**성과 지표:**
- 연환산 수익률: {metrics.get('annual_return', 0):.2f}%
- 변동성: {metrics.get('volatility', 0):.2f}%
- 샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}
- 최대 낙폭: {metrics.get('max_drawdown', 0):.2f}%
- 베타: {metrics.get('beta', 0):.2f}
- 알파: {metrics.get('alpha', 0):.2f}%

**보유 종목 (상위 5개):**
{chr(10).join([f"- {h['name']} ({h['ticker']}): {h['gain_loss_pct']:.2f}% 수익률" for h in holdings[:5]])}

이 포트폴리오에 대해 다음을 포함하여 전문적인 분석을 제공해주세요:
1. 전반적인 성과 평가 (좋은 점과 개선이 필요한 점)
2. 위험 대비 수익 분석
3. 포트폴리오 다각화 수준
4. 개선 제안사항

분석은 명확하고 실용적이며, 투자자가 이해하기 쉽게 작성해주세요.
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 전문 투자 분석가입니다. 포트폴리오 성과를 분석하고 실용적인 조언을 제공합니다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        analysis_result = response.choices[0].message.content

        print("✅ OpenAI API call successful")

        # Rate limit 업데이트
        ai_analysis_rate_limit[client_ip] = current_time

        # 캐시 저장
        ai_analysis_cache[client_ip] = {
            "cache_key": cache_key,
            "result": analysis_result,
            "timestamp": current_time,
        }

        return JSONResponse(
            content={"success": True, "analysis": analysis_result, "cached": False}
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        print("=" * 80)
        print("❌ ERROR in /api/ai-analysis endpoint:")
        print(error_trace)
        print("=" * 80)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"AI 분석 중 오류가 발생했습니다: {str(e)}",
            },
        )


if __name__ == "__main__":
    import uvicorn

    # 데이터베이스 초기화
    init_database()

    # Uvicorn으로 FastAPI 앱 실행
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
