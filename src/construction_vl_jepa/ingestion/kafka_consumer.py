"""Kafka consumer for streaming sensor data ingestion.

Consumes sensor readings from Kafka topics, validates schema,
and routes to the appropriate processing pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime

from construction_vl_jepa.config import KafkaConfig
from construction_vl_jepa.data.schema import IngestRecord, QualityFlag, SensorReading, SourceType
from construction_vl_jepa.utils.logging import get_logger

log = get_logger(__name__)


class SensorKafkaConsumer:
    """Consumes sensor readings from Kafka and yields validated records."""

    def __init__(self, cfg: KafkaConfig):
        self.cfg = cfg
        self.consumer = None

    def connect(self) -> None:
        """Initialize the Kafka consumer."""
        from confluent_kafka import Consumer

        self.consumer = Consumer({
            "bootstrap.servers": self.cfg.bootstrap_servers,
            "group.id": self.cfg.consumer_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        })
        self.consumer.subscribe([self.cfg.sensor_topic])
        log.info("kafka_consumer_connected", topic=self.cfg.sensor_topic)

    def consume_batch(self, max_messages: int = 100, timeout: float = 1.0) -> list[SensorReading]:
        """Consume a batch of sensor readings.

        Args:
            max_messages: Maximum messages to consume in one batch.
            timeout: Poll timeout in seconds.

        Returns:
            List of validated SensorReading objects.
        """
        if self.consumer is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        messages = self.consumer.consume(num_messages=max_messages, timeout=timeout)
        readings = []

        for msg in messages:
            if msg.error():
                log.warning("kafka_message_error", error=str(msg.error()))
                continue

            try:
                data = json.loads(msg.value().decode("utf-8"))
                reading = self._parse_reading(data)
                if reading:
                    readings.append(reading)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                log.warning("message_parse_error", error=str(e), raw=msg.value()[:200])
                # In production: send to dead letter queue

        return readings

    def _parse_reading(self, data: dict) -> SensorReading | None:
        """Parse and validate a raw message into a SensorReading."""
        try:
            return SensorReading(
                machine_id=data["machine_id"],
                sensor_id=data["sensor_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                value=float(data["value"]),
                quality_flag=QualityFlag(data.get("quality_flag", "good")),
            )
        except (KeyError, ValueError) as e:
            log.warning("invalid_reading", error=str(e))
            return None

    def close(self) -> None:
        """Close the consumer."""
        if self.consumer:
            self.consumer.close()
            log.info("kafka_consumer_closed")


class IngestRecordConsumer:
    """Generic consumer for the unified IngestRecord format."""

    def __init__(self, cfg: KafkaConfig, topic: str | None = None):
        self.cfg = cfg
        self.topic = topic or cfg.sensor_topic
        self.consumer = None

    def connect(self) -> None:
        from confluent_kafka import Consumer

        self.consumer = Consumer({
            "bootstrap.servers": self.cfg.bootstrap_servers,
            "group.id": f"{self.cfg.consumer_group}-ingest",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        })
        self.consumer.subscribe([self.topic])

    def consume_batch(self, max_messages: int = 100, timeout: float = 1.0) -> list[IngestRecord]:
        if self.consumer is None:
            raise RuntimeError("Consumer not connected")

        messages = self.consumer.consume(num_messages=max_messages, timeout=timeout)
        records = []

        for msg in messages:
            if msg.error():
                continue
            try:
                data = json.loads(msg.value().decode("utf-8"))
                record = IngestRecord(
                    machine_id=data["machine_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    source_type=SourceType(data["source_type"]),
                    payload=data["payload"],
                )
                records.append(record)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return records

    def close(self) -> None:
        if self.consumer:
            self.consumer.close()
