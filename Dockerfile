# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-gpu-images/python:v100

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install -U sklearn && \
    pip install fastprogress japanize-matplotlib && \
    pip install xfeat && \
    pip install nlp
