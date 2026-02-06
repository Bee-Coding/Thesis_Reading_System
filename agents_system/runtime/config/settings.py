"""
Thesis_Reading_System 全局配置
"""
import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum

# 加载.env文件
def load_dotenv():
    """从.env文件加载环境变量"""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # 移除引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ.setdefault(key, value)

# 在模块加载时自动加载.env
load_dotenv()


class ExecutionMode(Enum):
    """执行模式"""
    DEEP_INTERNALIZATION = "Deep_Internalization"
    QUICK_ASSESSMENT = "Quick_Assessment"
    ENGINEERING_REPRODUCTION = "Engineering_Reproduction"
    VALIDATION_FOCUSED = "Validation_Focused"


class AgentType(Enum):
    """Agent类型"""
    ORCHESTRATOR = "E2E-Learning-Orchestrator"
    SCHOLAR = "Scholar_Internalizer"
    CODE = "Code_Architect"
    VALIDATOR = "Scenario_Validator"
    VAULT = "Knowledge_Vault"
    CRITIC = "Strategic_Critic"


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 8000
    temperature: float = 0.3
    timeout: int = 300
    
    def __post_init__(self):
        # 从环境变量读取配置
        self.provider = os.getenv("LLM_PROVIDER", self.provider)
        self.model = os.getenv("LLM_MODEL", self.model)
        
        # 从环境变量读取API密钥
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
                self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "azure":
                self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
                self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "thesis_reading_system"
    user: str = "postgres"
    password: Optional[str] = None
    
    def __post_init__(self):
        # 从环境变量读取配置
        self.host = os.getenv("DB_HOST", self.host)
        self.port = int(os.getenv("DB_PORT", str(self.port)))
        self.database = os.getenv("DB_NAME", self.database)
        self.user = os.getenv("DB_USER", self.user)
        if self.password is None:
            self.password = os.getenv("DB_PASSWORD", "")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class QualityGateConfig:
    """质量门配置"""
    default_threshold: str = "B"  # A, B, C, D
    min_quality_score: float = 0.7
    enable_pre_execution: bool = True
    enable_post_execution: bool = True
    enable_integration: bool = True
    max_retry_on_failure: int = 2


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay_seconds: int = 5
    max_delay_seconds: int = 300
    backoff_strategy: str = "exponential"  # fixed, linear, exponential
    retryable_errors: List[str] = field(default_factory=lambda: [
        "E001", "E002", "E003", "E004", "EA02", "ED01"
    ])


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 3
    reset_timeout_seconds: int = 300
    half_open_max_attempts: int = 1


@dataclass
class Mem0Config:
    """Mem0记忆管理配置"""
    api_key: Optional[str] = None
    enabled: bool = True
    user_id: str = "zhn"
    organization_id: str = "thesis_reading_system"
    auto_save: bool = True
    context_window: int = 10
    relevance_threshold: float = 0.7
    
    def __post_init__(self):
        # 从环境变量读取配置
        self.api_key = os.getenv("MEM0_API_KEY", self.api_key)
        self.enabled = os.getenv("MEM0_ENABLED", str(self.enabled)).lower() == "true"
        self.user_id = os.getenv("MEM0_USER_ID", self.user_id)
        self.organization_id = os.getenv("MEM0_ORGANIZATION_ID", self.organization_id)
        self.auto_save = os.getenv("MEMORY_AUTO_SAVE", str(self.auto_save)).lower() == "true"
        self.context_window = int(os.getenv("MEMORY_CONTEXT_WINDOW", str(self.context_window)))
        self.relevance_threshold = float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", str(self.relevance_threshold)))


@dataclass
class Settings:
    """全局设置"""
    # 基础路径 - 指向项目根目录
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    
    # 子模块配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    mem0: Mem0Config = field(default_factory=Mem0Config)
    
    # 目录配置
    @property
    def agents_dir(self) -> Path:
        return self.base_dir / "agents_system"
    
    @property
    def protocols_dir(self) -> Path:
        return self.agents_dir / "protocols"
    
    @property
    def schemas_dir(self) -> Path:
        return self.protocols_dir / "schemas"
    
    @property
    def atoms_dir(self) -> Path:
        return self.base_dir / "atoms"
    
    @property
    def reports_dir(self) -> Path:
        return self.base_dir / "reports"
    
    @property
    def manifests_dir(self) -> Path:
        return self.base_dir / "manifests"
    
    @property
    def raw_papers_dir(self) -> Path:
        return self.base_dir / "raw_papers"
    
    # Agent Prompt文件路径
    def get_agent_prompt_path(self, agent_type: AgentType) -> Path:
        agent_files = {
            AgentType.ORCHESTRATOR: "E2E-Learning-Orchestrator.md",
            AgentType.SCHOLAR: "Scholar_Internalizer.md",
            AgentType.CODE: "Code_Architect.md",
            AgentType.VALIDATOR: "Scenario_Validator.md",
            AgentType.VAULT: "Knowledge_Vault.md",
            AgentType.CRITIC: "Strategic_Critic.md",
        }
        return self.agents_dir / agent_files[agent_type]
    
    def get_agent_prompt(self, agent_type: AgentType) -> str:
        """读取Agent的System Prompt"""
        prompt_path = self.get_agent_prompt_path(agent_type)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Agent prompt not found: {prompt_path}")


# 全局设置实例
settings = Settings()


# Agent特定的LLM参数配置
AGENT_LLM_PARAMS = {
    AgentType.SCHOLAR: {
        "temperature": 0.25,
        "max_tokens": 12000,
        "timeout": 600,
    },
    AgentType.CODE: {
        "temperature": 0.15,
        "max_tokens": 15000,
        "timeout": 600,
    },
    AgentType.VALIDATOR: {
        "temperature": 0.35,
        "max_tokens": 8000,
        "timeout": 300,
    },
    AgentType.VAULT: {
        "temperature": 0.1,
        "max_tokens": 4000,
        "timeout": 120,
    },
    AgentType.CRITIC: {
        "temperature": 0.4,
        "max_tokens": 8000,
        "timeout": 300,
    },
}


# 执行模式对应的参数调整
MODE_ADJUSTMENTS = {
    ExecutionMode.DEEP_INTERNALIZATION: {
        "temperature_delta": 0,
        "max_tokens_multiplier": 1.5,
        "timeout_multiplier": 2,
    },
    ExecutionMode.QUICK_ASSESSMENT: {
        "temperature_delta": 0.1,
        "max_tokens_multiplier": 0.5,
        "timeout_multiplier": 0.5,
    },
    ExecutionMode.ENGINEERING_REPRODUCTION: {
        "temperature_delta": -0.1,
        "max_tokens_multiplier": 1.2,
        "timeout_multiplier": 1.5,
    },
    ExecutionMode.VALIDATION_FOCUSED: {
        "temperature_delta": 0.05,
        "max_tokens_multiplier": 1.0,
        "timeout_multiplier": 1.0,
    },
}
