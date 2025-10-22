"""
포트폴리오 테이블에 user_id 컬럼 추가 및 기존 데이터 정리
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "stock_cache.db")


def migrate_portfolio_table():
    """포트폴리오 테이블에 user_id 추가 및 기존 데이터 삭제"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # saved_portfolios 테이블이 존재하는지 확인
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='saved_portfolios'
        """
        )

        if not cursor.fetchone():
            print("❌ saved_portfolios 테이블이 존재하지 않습니다.")
            return False

        # user_id 컬럼이 이미 존재하는지 확인
        cursor.execute("PRAGMA table_info(saved_portfolios)")
        columns = [column[1] for column in cursor.fetchall()]

        if "user_id" in columns:
            print("✅ user_id 컬럼이 이미 존재합니다.")
        else:
            # user_id 컬럼 추가
            print("📝 user_id 컬럼을 추가하는 중...")
            cursor.execute(
                """
                ALTER TABLE saved_portfolios 
                ADD COLUMN user_id INTEGER
            """
            )
            print("✅ user_id 컬럼 추가 완료")

        # 기존 데이터 삭제 (user_id가 NULL인 데이터)
        print("\n📝 user_id가 NULL인 기존 포트폴리오 데이터를 삭제하는 중...")
        cursor.execute("SELECT COUNT(*) FROM saved_portfolios WHERE user_id IS NULL")
        count = cursor.fetchone()[0]

        if count > 0:
            print(f"   삭제할 레코드: {count}개")
            cursor.execute("DELETE FROM saved_portfolios WHERE user_id IS NULL")
            print(f"✅ {count}개의 기존 포트폴리오 데이터 삭제 완료")
        else:
            print("   삭제할 데이터가 없습니다.")

        conn.commit()

        # 결과 확인
        cursor.execute("PRAGMA table_info(saved_portfolios)")
        print("\n📋 현재 saved_portfolios 테이블 구조:")
        for column in cursor.fetchall():
            print(f"   - {column[1]}: {column[2]}")

        cursor.execute("SELECT COUNT(*) FROM saved_portfolios")
        remaining = cursor.fetchone()[0]
        print(f"\n📊 남은 포트폴리오 수: {remaining}개")

        return True

    except Exception as e:
        print(f"❌ 마이그레이션 중 오류 발생: {e}")
        if conn:
            conn.rollback()
        return False

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 포트폴리오 테이블 마이그레이션 시작")
    print("=" * 60)
    print(f"DB 경로: {DB_PATH}\n")

    if migrate_portfolio_table():
        print("\n" + "=" * 60)
        print("✅ 마이그레이션이 성공적으로 완료되었습니다!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 마이그레이션 실패")
        print("=" * 60)
