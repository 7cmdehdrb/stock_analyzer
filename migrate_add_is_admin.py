import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "stock_cache.db")


def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # is_admin 컬럼이 있는지 확인
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]

        if "is_admin" not in columns:
            print("is_admin 컬럼을 추가합니다...")

            # is_admin 컬럼 추가 (기본값 False)
            cursor.execute(
                """
                ALTER TABLE users 
                ADD COLUMN is_admin INTEGER DEFAULT 0 NOT NULL
            """
            )

            conn.commit()
            print("✅ is_admin 컬럼이 성공적으로 추가되었습니다.")

            # 현재 사용자 수 확인
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            print(f"   총 {user_count}명의 사용자에게 is_admin = False 설정 완료")

        else:
            print("⚠️  is_admin 컬럼이 이미 존재합니다.")

        # 테이블 구조 확인
        print("\n현재 users 테이블 구조:")
        cursor.execute("PRAGMA table_info(users)")
        for column in cursor.fetchall():
            print(
                f"  - {column[1]}: {column[2]} (Nullable: {not column[3]}, Default: {column[4]})"
            )

    except Exception as e:
        print(f"❌ 마이그레이션 중 오류 발생: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
