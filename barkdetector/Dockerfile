# https://developers.home-assistant.io/docs/apps/configuration#app-dockerfile
ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-debian:bookworm-2026.03.0
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libportaudio2   -> sounddevice's backend
# libasound2-plugins -> ALSA "pulse" plugin, how PortAudio reaches the
#                      Home Assistant audio plugin's PulseAudio server
# alsa-utils      -> arecord/aplay, for debugging the mic from the add-on shell
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libportaudio2 \
        libasound2-plugins \
        alsa-utils \
        pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt

# --break-system-packages: Debian marks the system interpreter as externally
# managed (PEP 668); inside a single-purpose container installing into it is
# what we want, and a venv would only add a layer of indirection.
RUN python3 -m pip install --break-system-packages -r /app/requirements.txt

COPY app/ /app/
COPY rootfs /

RUN chmod a+x /etc/services.d/barkdetector/run \
    && chmod a+x /etc/services.d/barkdetector/finish \
    && chmod a+x /usr/bin/render_config.py

LABEL \
    org.opencontainers.image.title="Home Assistant App: Bark Detector" \
    org.opencontainers.image.description="Dog bark detection from a local USB microphone via YAMNet, published to MQTT." \
    org.opencontainers.image.source="https://github.com/mauriciomorales/barkdetector" \
    org.opencontainers.image.licenses="MIT"
