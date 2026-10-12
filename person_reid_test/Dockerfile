# Independent first-phase ReID image. It does not modify the robot runtime image.
FROM wpb-owner-search:noetic

# Official Torchreid source; import only its model package to avoid training-only dependencies.
RUN git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git /opt/torchreid && sed -i 's/from torchreid import data, optim, utils, engine, losses, models, metrics/from torchreid import models/' /opt/torchreid/torchreid/__init__.py

WORKDIR /workspace/person_reid_test
ENV PYTHONPATH=/opt/torchreid
ENV MPLBACKEND=Agg
