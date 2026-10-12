FROM osrf/ros:noetic-desktop-full

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=ubuntu20
ARG USER_UID=1000
ARG USER_GID=1000
ENV ROS_DISTRO=noetic \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/home/ubuntu20 \
    USER=ubuntu20 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ULTRALYTICS_CONFIG_DIR=/home/ubuntu20/.config/Ultralytics

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    flite \
    git \
    gnupg2 \
    libasound2-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglfw3-dev \
    libjpeg-dev \
    libopenblas-dev \
    libopenni-dev \
    libopenni2-dev \
    libportaudio2 \
    libpulse0 \
    libtinyxml-dev \
    libturbojpeg0-dev \
    libusb-1.0-0-dev \
    libv4l-dev \
    locales \
    pkg-config \
    portaudio19-dev \
    python3-dev \
    python3-empy \
    python3-numpy \
    python3-opencv \
    python3-pip \
    python3-rosdep \
    python3-rospkg \
    python3-yaml \
    ros-noetic-amcl \
    ros-noetic-cmake-modules \
    ros-noetic-compressed-depth-image-transport \
    ros-noetic-controller-manager \
    ros-noetic-depth-image-proc \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-image-view \
    ros-noetic-joint-state-controller \
    ros-noetic-map-server \
    ros-noetic-move-base \
    ros-noetic-navigation \
    ros-noetic-nodelet \
    ros-noetic-rplidar-ros \
    ros-noetic-robot-state-publisher \
    ros-noetic-sound-play \
    ros-noetic-tf \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-ros \
    ros-noetic-xacro \
    sudo \
    udev \
    && rm -rf /var/lib/apt/lists/*

RUN if ! getent group ${USER_GID} >/dev/null; then groupadd --gid ${USER_GID} ${USERNAME}; fi \
    && if ! id -u ${USERNAME} >/dev/null 2>&1; then useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}; fi \
    && usermod -aG sudo,dialout,audio,video,plugdev ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

COPY requirements-robot.txt /tmp/requirements-robot.txt
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install tokenizers==0.20.3
RUN python3 -m pip install --ignore-installed psutil==7.2.2
RUN python3 -m pip install -r /tmp/requirements-robot.txt
RUN python3 -m pip install --no-deps albumentations==1.3.1 qudida==0.0.4 insightface==0.7.3

# ROS Noetic remains on Python 3.8. RTX 50-series GPUs require a newer
# PyTorch CUDA binary, whose CUDA 12.8 wheels start at Python 3.9. Keep the
# ROS/runtime dependencies on Python 3.8 and give the YOLO node its own
# Python 3.9 environment; yoloworld_node.py does not load cv_bridge so it is
# safe to run it with this interpreter.
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.9 python3.9-dev python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://bootstrap.pypa.io/pip/3.9/get-pip.py -o /tmp/get-pip.py \
    && python3.9 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && python3.9 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
         torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    && python3.9 -m pip install --no-cache-dir --ignore-installed psutil==7.2.2 \
    && python3.9 -m pip install --no-cache-dir \
         numpy==1.24.4 opencv-python-headless==4.10.0.84 \
         rospkg catkin_pkg empy defusedxml netifaces psutil==7.2.2 \
         ultralytics==8.4.118 ultralytics-thop==2.1.6 \
    && python3.9 -m pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git@488e81a6711eea7346872b46ea928b367da8889d

COPY libfreenect2 /tmp/libfreenect2-src
COPY freenect2 /home/ubuntu20/freenect2
RUN if [ -d /tmp/libfreenect2-src ]; then \
      cmake -S /tmp/libfreenect2-src -B /tmp/libfreenect2-build \
        -DCMAKE_INSTALL_PREFIX=/home/ubuntu20/freenect2 \
        -DENABLE_OPENCL=OFF \
        -DENABLE_OPENGL=OFF \
        -DENABLE_CUDA=OFF \
        -DBUILD_EXAMPLES=OFF \
      && cmake --build /tmp/libfreenect2-build --target install -- -j"$(nproc)"; \
    fi \
    && chown -R ${USERNAME}:${USERNAME} /home/ubuntu20/freenect2

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /home/ubuntu20/catkin_ws/src /home/ubuntu20/.insightface /home/ubuntu20/weights /home/ubuntu20/.config/Ultralytics \
    && chown -R ${USERNAME}:${USERNAME} /home/ubuntu20/catkin_ws /home/ubuntu20/.config
RUN apt-get update \
    && apt-get install -y --no-install-recommends ros-noetic-joy \
    && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
WORKDIR /home/ubuntu20/catkin_ws
RUN printf '%s\n' \
       'source /opt/ros/noetic/setup.bash' \
       'if [ -f /home/ubuntu20/catkin_ws/devel/setup.bash ]; then source /home/ubuntu20/catkin_ws/devel/setup.bash; fi' \
       'export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}' \
       >> /home/ubuntu20/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# Keep the CUDA/PyTorch build layer cached. InsightFace 0.7.3 imports a
# Matplotlib API removed in newer releases, so pin its Python 3.9 runtime.
USER root
RUN python3.9 -m pip install --no-cache-dir --force-reinstall \
      matplotlib==3.7.5 albumentations==1.3.1 qudida==0.0.4 insightface==0.7.3

# Add Qwen after all existing ROS/CUDA layers so routine Qwen updates reuse
# the already-built robot runtime. The task calls it on localhost:11434.
ARG OLLAMA_MODEL=qwen3.5:0.8b
ENV OLLAMA_MODELS=/opt/ollama/models
RUN curl -fsSL https://ollama.com/install.sh -o /tmp/install-ollama.sh \
    && sh /tmp/install-ollama.sh \
    && command -v ollama \
    && rm -f /tmp/install-ollama.sh \
    && mkdir -p "${OLLAMA_MODELS}" \
    && (ollama serve >/tmp/ollama-build.log 2>&1 & \
        ollama_pid=$!; \
        for attempt in $(seq 1 30); do \
          if ollama list >/dev/null 2>&1; then break; fi; \
          sleep 1; \
        done; \
        ollama list >/dev/null 2>&1; \
        ollama pull "${OLLAMA_MODEL}"; \
        kill "${ollama_pid}"; \
        wait "${ollama_pid}" 2>/dev/null || true)
USER ${USERNAME}
