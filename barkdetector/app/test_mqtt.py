#!/usr/bin/env python3
"""Test MQTT connection and subscribe to bark events."""

import paho.mqtt.client as mqtt
import os
import time
import json
import sys

BROKER_HOST = os.environ.get("BARKDETECTOR_MQTT_HOST", "127.0.0.1")
BROKER_PORT = int(os.environ.get("BARKDETECTOR_MQTT_PORT", "1883"))
BROKER_USERNAME = os.environ.get("BARKDETECTOR_MQTT_USERNAME", "")
BROKER_PASSWORD = os.environ.get("BARKDETECTOR_MQTT_PASSWORD", "")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker successfully!")
        print("🎧 Subscribing to topic: home/sensors/dog_bark")
        client.subscribe("home/sensors/dog_bark")
    else:
        print(f"❌ Connection failed with code: {rc}")
        print("   0: Success")
        print("   1: Incorrect protocol version")
        print("   2: Invalid client ID")
        print("   3: Server unavailable")
        print("   4: Bad username or password")
        print("   5: Not authorized")

def on_message(client, userdata, msg):
    print(f"\n🐶 BARK EVENT RECEIVED!")
    print(f"   Topic: {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        print(f"   Payload: {json.dumps(payload, indent=2)}")
    except:
        print(f"   Payload: {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"⚠️  Unexpected disconnection (code: {rc})")

def cleanup(client):
    """Properly cleanup MQTT client and connections."""
    try:
        print("\n🧹 Cleaning up...")
        client.loop_stop()
        client.disconnect()
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

# Create MQTT client with unique ID to avoid conflicts
client = mqtt.Client(client_id="barkdetector-test", clean_session=True)
if BROKER_USERNAME:
    client.username_pw_set(BROKER_USERNAME, BROKER_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

print(f"🔌 Connecting to MQTT broker at {BROKER_HOST}:{BROKER_PORT}...")
print("   Client ID: barkdetector-test")
try:
    client.connect(BROKER_HOST, BROKER_PORT, 60)
    client.loop_start()

    print("👂 Listening for bark events... (Press Ctrl+C to stop)")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n👋 Stopping...")
    cleanup(client)
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    cleanup(client)
    sys.exit(1)
