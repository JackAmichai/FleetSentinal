"""
FleetSentinel Configuration Module

Central configuration management with environment variable support.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class Environment(Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    
    # Pool settings
    max_connections: int = 100
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    @property
    def url(self) -> str:
        """Build Redis URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
    
    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "100")),
        )


@dataclass
class KafkaConfig:
    """Kafka connection configuration."""
    bootstrap_servers: str = "localhost:9092"
    topic_ride_requests: str = "fleet.ride.requests"
    topic_threat_alerts: str = "fleet.threat.alerts"
    topic_decisions: str = "fleet.decisions"
    
    # Consumer settings
    consumer_group: str = "fleet-sentinel"
    auto_offset_reset: str = "latest"
    
    # Producer settings
    acks: str = "all"
    retries: int = 3
    
    @classmethod
    def from_env(cls) -> "KafkaConfig":
        """Load configuration from environment variables."""
        return cls(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic_ride_requests=os.getenv("KAFKA_TOPIC_REQUESTS", "fleet.ride.requests"),
            topic_threat_alerts=os.getenv("KAFKA_TOPIC_ALERTS", "fleet.threat.alerts"),
            topic_decisions=os.getenv("KAFKA_TOPIC_DECISIONS", "fleet.decisions"),
            consumer_group=os.getenv("KAFKA_CONSUMER_GROUP", "fleet-sentinel"),
        )


@dataclass
class StripeConfig:
    """Stripe API configuration."""
    api_key: str = ""
    webhook_secret: str = ""
    
    # Hold settings
    default_hold_amount_cents: int = 50000  # $500
    hold_expiration_minutes: int = 60
    
    @classmethod
    def from_env(cls) -> "StripeConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("STRIPE_API_KEY", ""),
            webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET", ""),
            default_hold_amount_cents=int(os.getenv("STRIPE_HOLD_AMOUNT_CENTS", "50000")),
            hold_expiration_minutes=int(os.getenv("STRIPE_HOLD_EXPIRATION_MINUTES", "60")),
        )


@dataclass
class SentinelConfig:
    """Core sentinel detection configuration."""
    # Time window for density analysis
    time_window_seconds: int = 300
    
    # Geospatial clustering
    cluster_radius_meters: float = 200
    min_cluster_size: int = 5
    
    # Confidence thresholds
    confidence_liability_threshold: float = 0.4
    confidence_block_threshold: float = 0.8
    
    # Feature weights for confidence calculation
    density_weight: float = 0.4
    velocity_weight: float = 0.3
    topology_weight: float = 0.2
    user_risk_weight: float = 0.1
    
    # Topology risk multipliers
    dead_end_multiplier: float = 2.5
    cul_de_sac_multiplier: float = 2.0
    residential_multiplier: float = 1.5
    commercial_multiplier: float = 1.2
    arterial_multiplier: float = 0.2
    highway_multiplier: float = 0.1
    
    # User risk factors
    new_account_threshold_days: int = 7
    low_ride_count_threshold: int = 5
    
    @classmethod
    def from_env(cls) -> "SentinelConfig":
        """Load configuration from environment variables."""
        return cls(
            time_window_seconds=int(os.getenv("SENTINEL_TIME_WINDOW_SECONDS", "300")),
            cluster_radius_meters=float(os.getenv("SENTINEL_CLUSTER_RADIUS_METERS", "200")),
            min_cluster_size=int(os.getenv("SENTINEL_MIN_CLUSTER_SIZE", "5")),
            confidence_liability_threshold=float(os.getenv("SENTINEL_LIABILITY_THRESHOLD", "0.4")),
            confidence_block_threshold=float(os.getenv("SENTINEL_BLOCK_THRESHOLD", "0.8")),
            density_weight=float(os.getenv("SENTINEL_DENSITY_WEIGHT", "0.4")),
            velocity_weight=float(os.getenv("SENTINEL_VELOCITY_WEIGHT", "0.3")),
            topology_weight=float(os.getenv("SENTINEL_TOPOLOGY_WEIGHT", "0.2")),
            user_risk_weight=float(os.getenv("SENTINEL_USER_RISK_WEIGHT", "0.1")),
        )


@dataclass
class ContextOracleConfig:
    """Context verification service configuration."""
    # Event APIs
    ticketmaster_api_key: str = ""
    eventbrite_api_key: str = ""
    
    # Civic data APIs
    civic_data_api_key: str = ""
    
    # Cache settings
    event_cache_ttl_seconds: int = 300  # 5 minutes
    civic_cache_ttl_seconds: int = 60   # 1 minute
    
    # Thresholds
    event_radius_meters: float = 500
    crowd_threshold: int = 1000
    
    @classmethod
    def from_env(cls) -> "ContextOracleConfig":
        """Load configuration from environment variables."""
        return cls(
            ticketmaster_api_key=os.getenv("TICKETMASTER_API_KEY", ""),
            eventbrite_api_key=os.getenv("EVENTBRITE_API_KEY", ""),
            civic_data_api_key=os.getenv("CIVIC_DATA_API_KEY", ""),
            event_cache_ttl_seconds=int(os.getenv("EVENT_CACHE_TTL_SECONDS", "300")),
            civic_cache_ttl_seconds=int(os.getenv("CIVIC_CACHE_TTL_SECONDS", "60")),
            event_radius_meters=float(os.getenv("EVENT_RADIUS_METERS", "500")),
        )


@dataclass
class VehicleDefenseConfig:
    """Vehicle self-defense module configuration."""
    # Turtle mode
    turtle_speed_mph: float = 0.5
    turtle_gap_meters: float = 0.5
    
    # Vandalism guard
    max_escalation_level: int = 3
    auto_deescalate_seconds: int = 300  # 5 minutes
    
    # Escape planning
    escape_route_radius_meters: float = 500
    min_safe_distance_meters: float = 100
    
    # Threat detection model
    model_path: str = "models/threat_detector.pt"
    inference_threshold: float = 0.7
    
    @classmethod
    def from_env(cls) -> "VehicleDefenseConfig":
        """Load configuration from environment variables."""
        return cls(
            turtle_speed_mph=float(os.getenv("DEFENSE_TURTLE_SPEED_MPH", "0.5")),
            turtle_gap_meters=float(os.getenv("DEFENSE_TURTLE_GAP_METERS", "0.5")),
            max_escalation_level=int(os.getenv("DEFENSE_MAX_ESCALATION", "3")),
            auto_deescalate_seconds=int(os.getenv("DEFENSE_AUTO_DEESCALATE_SECONDS", "300")),
            escape_route_radius_meters=float(os.getenv("DEFENSE_ESCAPE_RADIUS_METERS", "500")),
            model_path=os.getenv("DEFENSE_MODEL_PATH", "models/threat_detector.pt"),
            inference_threshold=float(os.getenv("DEFENSE_INFERENCE_THRESHOLD", "0.7")),
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Structured logging
    json_format: bool = True
    
    # Log destinations
    console: bool = True
    file_path: Optional[str] = None
    
    # External logging services
    datadog_api_key: Optional[str] = None
    sentry_dsn: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Load configuration from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            json_format=os.getenv("LOG_JSON_FORMAT", "true").lower() == "true",
            console=os.getenv("LOG_CONSOLE", "true").lower() == "true",
            file_path=os.getenv("LOG_FILE_PATH"),
            datadog_api_key=os.getenv("DATADOG_API_KEY"),
            sentry_dsn=os.getenv("SENTRY_DSN"),
        )


@dataclass
class FleetSentinelConfig:
    """Master configuration for FleetSentinel."""
    environment: Environment = Environment.DEVELOPMENT
    
    # Sub-configurations
    redis: RedisConfig = field(default_factory=RedisConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    stripe: StripeConfig = field(default_factory=StripeConfig)
    sentinel: SentinelConfig = field(default_factory=SentinelConfig)
    context_oracle: ContextOracleConfig = field(default_factory=ContextOracleConfig)
    vehicle_defense: VehicleDefenseConfig = field(default_factory=VehicleDefenseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_env(cls) -> "FleetSentinelConfig":
        """Load all configuration from environment variables."""
        env_str = os.getenv("ENVIRONMENT", "development").lower()
        try:
            environment = Environment(env_str)
        except ValueError:
            environment = Environment.DEVELOPMENT
        
        return cls(
            environment=environment,
            redis=RedisConfig.from_env(),
            kafka=KafkaConfig.from_env(),
            stripe=StripeConfig.from_env(),
            sentinel=SentinelConfig.from_env(),
            context_oracle=ContextOracleConfig.from_env(),
            vehicle_defense=VehicleDefenseConfig.from_env(),
            logging=LoggingConfig.from_env(),
        )
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration and return any warnings/errors.
        
        Returns:
            Dictionary with 'valid' boolean and 'messages' list
        """
        messages = []
        valid = True
        
        # Check Redis
        if not self.redis.host:
            messages.append("WARNING: Redis host not configured")
        
        # Check Stripe in production
        if self.environment == Environment.PRODUCTION:
            if not self.stripe.api_key:
                messages.append("ERROR: Stripe API key required in production")
                valid = False
            if not self.stripe.api_key.startswith("sk_live_"):
                messages.append("WARNING: Using non-live Stripe key in production")
        
        # Check Kafka
        if self.environment == Environment.PRODUCTION:
            if self.kafka.bootstrap_servers == "localhost:9092":
                messages.append("WARNING: Using localhost Kafka in production")
        
        return {"valid": valid, "messages": messages}


# Global configuration instance
_config: Optional[FleetSentinelConfig] = None


def get_config() -> FleetSentinelConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = FleetSentinelConfig.from_env()
    return _config


def set_config(config: FleetSentinelConfig) -> None:
    """Set the global configuration instance (for testing)."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration (for testing)."""
    global _config
    _config = None
