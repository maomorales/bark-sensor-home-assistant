"""MQTT publishing helper."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from paho.mqtt import client as mqtt


@dataclass
class MQTTConfig:
    host: str
    port: int
    topic: str
    username: str | None = None
    password: str | None = None


class MQTTPublisher:
    """Publish events to MQTT with automatic reconnection."""

    def __init__(self, config: MQTTConfig, client_id: Optional[str] = None) -> None:
        self.config = config
        self.client = mqtt.Client(client_id=client_id or "", clean_session=True)
        if config.username:
            self.client.username_pw_set(config.username, config.password or "")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._connected = threading.Event()
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def start(self) -> None:
        """Connect to the broker and start the network loop."""
        try:
            self.client.connect(self.config.host, int(self.config.port))
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Initial MQTT connection failed: {}", exc)
        self.client.loop_start()

    def stop(self) -> None:
        """Stop the loop and disconnect."""
        self.client.loop_stop()
        try:
            self.client.disconnect()
        except Exception:
            pass

    def publish(self, payload: dict, qos: int = 0, retain: bool = False) -> None:
        """Publish a JSON payload to the configured topic."""
        data = json.dumps(payload)
        if not self._connected.wait(timeout=2.0):
            logger.warning("MQTT client not connected; attempting publish anyway")
        result = self.client.publish(self.config.topic, data, qos=qos, retain=retain)
        if result.rc not in (mqtt.MQTT_ERR_SUCCESS, mqtt.MQTT_ERR_NO_CONN):
            logger.error("MQTT publish failed with code {}", result.rc)

    # Callbacks -------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):  # type: ignore[override]
        if rc == 0:
            logger.info("Connected to MQTT broker at {}:{}", self.config.host, self.config.port)
            self._connected.set()
        else:
            logger.error("MQTT connection refused (code {})", rc)

    def _on_disconnect(self, client, userdata, rc):  # type: ignore[override]
        if rc != 0:
            logger.warning("Unexpected MQTT disconnection (code {}), retrying", rc)
            # give some time before declaring offline to avoid thrashing
            time.sleep(1.0)
        self._connected.clear()
