import time
import json
import logging
import sys
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs as structured JSON objects."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Inject extra telemetry attributes if present
        if hasattr(record, "req_method"):
            log_record["method"] = record.req_method
        if hasattr(record, "req_url"):
            log_record["url"] = record.req_url
        if hasattr(record, "status_code"):
            log_record["status_code"] = record.status_code
        if hasattr(record, "duration_ms"):
            log_record["duration_ms"] = record.duration_ms
            
        return json.dumps(log_record)


def setup_json_logging():
    """Configures root logger to output structured JSON streams."""
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


class TelemetryLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that measures request execution time and logs structured telemetry."""
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        
        # Proceed with request handling
        response = await call_next(request)
        
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        logger = logging.getLogger("uvicorn.access")
        logger.info(
            f"HTTP request processed",
            extra={
                "req_method": request.method,
                "req_url": str(request.url),
                "status_code": response.status_code,
                "duration_ms": duration_ms
            }
        )
        return response