"""
数据库连接模块 - 管理PostgreSQL数据库连接
"""
import asyncio
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from ..config.settings import settings, DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """数据库连接管理器"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or settings.database
        self._pool: Optional[Any] = None
        self._sync_conn: Optional[Any] = None
    
    async def connect(self) -> None:
        """建立异步连接池"""
        if not HAS_ASYNCPG:
            logger.warning("asyncpg not installed, async operations unavailable")
            return
        
        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info(f"Connected to database: {self.config.database}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self) -> None:
        """关闭连接池"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")
    
    def connect_sync(self) -> None:
        """建立同步连接"""
        if not HAS_PSYCOPG2:
            logger.warning("psycopg2 not installed, sync operations unavailable")
            return
        
        try:
            self._sync_conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            logger.info(f"Sync connected to database: {self.config.database}")
        except Exception as e:
            logger.error(f"Failed to sync connect to database: {e}")
            raise
    
    def disconnect_sync(self) -> None:
        """关闭同步连接"""
        if self._sync_conn:
            self._sync_conn.close()
            self._sync_conn = None
            logger.info("Sync database connection closed")
    
    @asynccontextmanager
    async def acquire(self):
        """获取连接（异步上下文管理器）"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args) -> str:
        """执行SQL语句"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """查询多行数据"""
        async with self.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """查询单行数据"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetchval(self, query: str, *args) -> Any:
        """查询单个值"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    def execute_sync(self, query: str, params: tuple = None) -> None:
        """同步执行SQL语句"""
        if not self._sync_conn:
            self.connect_sync()
        
        with self._sync_conn.cursor() as cur:
            cur.execute(query, params)
            self._sync_conn.commit()
    
    def fetch_sync(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """同步查询多行数据"""
        if not self._sync_conn:
            self.connect_sync()
        
        with self._sync_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()
    
    def fetchone_sync(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """同步查询单行数据"""
        if not self._sync_conn:
            self.connect_sync()
        
        with self._sync_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchone()


# 全局连接实例
_connection: Optional[DatabaseConnection] = None


def get_connection() -> DatabaseConnection:
    """获取全局数据库连接实例"""
    global _connection
    if _connection is None:
        _connection = DatabaseConnection()
    return _connection


async def init_database() -> DatabaseConnection:
    """初始化数据库连接"""
    conn = get_connection()
    await conn.connect()
    return conn


async def close_database() -> None:
    """关闭数据库连接"""
    global _connection
    if _connection:
        await _connection.disconnect()
        _connection = None
