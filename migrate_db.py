"""
데이터베이스 마이그레이션 스크립트
기존 users 테이블에 account_type 컬럼 추가
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "stock_cache.db")


def migrate_database():
    """기존 데이터베이스에 새 컬럼 추가"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # users 테이블이 존재하는지 확인
        cursor.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='users'
        """
        )

        if not cursor.fetchone():
            print("❌ users 테이블이 존재하지 않습니다.")
            print("   app.py를 실행하여 테이블을 먼저 생성하세요.")
            return False

        # account_type 컬럼이 이미 존재하는지 확인
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]

        if "account_type" in columns:
            print("✅ account_type 컬럼이 이미 존재합니다.")
            return True

        # account_type 컬럼 추가
        print("📝 account_type 컬럼을 추가하는 중...")
        cursor.execute(
            """
            ALTER TABLE users 
            ADD COLUMN account_type VARCHAR(20) DEFAULT 'local' NOT NULL
        """
        )

        # password_hash를 nullable로 변경 (SQLite는 ALTER COLUMN을 지원하지 않음)
        # 대신 기존 데이터는 모두 local 계정이므로 password_hash가 있을 것

        conn.commit()
        print("✅ 데이터베이스 마이그레이션 완료!")
        print("   - account_type 컬럼 추가됨 (기본값: 'local')")

        # 결과 확인
        cursor.execute("PRAGMA table_info(users)")
        print("\n📋 현재 users 테이블 구조:")
        for column in cursor.fetchall():
            print(f"   - {column[1]}: {column[2]}")

        return True

    except sqlite3.OperationalError as e:
        print(f"❌ 마이그레이션 중 오류 발생: {e}")
        if conn:
            conn.rollback()
        return False

    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        if conn:
            conn.rollback()
        return False

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 데이터베이스 마이그레이션 시작")
    print("=" * 60)
    print(f"DB 경로: {DB_PATH}\n")

    if migrate_database():
        print("\n" + "=" * 60)
        print("✅ 마이그레이션이 성공적으로 완료되었습니다!")
        print("=" * 60)
        print("\n이제 Flask 앱을 실행하고 /admin/user/ 에 접속하세요.")
    else:
        print("\n" + "=" * 60)
        print("❌ 마이그레이션 실패")
        print("=" * 60)
        print("\n해결 방법:")
        print("1. 테스트 환경이라면 stock_cache.db 파일을 삭제하고 다시 시작")
        print("2. 또는 아래 명령으로 테이블 재생성:")
        print(
            '   python -c "from app import app, db; app.app_context().push(); db.drop_all(); db.create_all()"'
        )
