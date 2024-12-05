FROM python:3.8-slim-buster

ENV DEBIAN_FRONTEND noninteractive





RUN apt-get update && \
    apt-get install -yqq apt-utils make gcc g++ build-essential ca-certificates libpq-dev curl nano wget


ARG PREFIX=/home/r/.local


RUN mkdir /app
WORKDIR /app



# TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure && \
#--prefix=$PREFIX && \
  make && \
  make install




ARG WEBAPP_USER_UID=${WEBAPP_USER_UID:-1000}
ARG WEBAPP_USER_GID=${WEBAPP_USER_GID:-${WEBAPP_USER_UID}}

RUN groupadd -f --gid ${WEBAPP_USER_GID} r && \
    useradd -u ${WEBAPP_USER_UID} -g ${WEBAPP_USER_GID} -m r


RUN TA_LIBRARY_PATH=$PREFIX/lib
RUN TA_INCLUDE_PATH=$PREFIX/include

RUN python -m pip install --no-cache-dir --upgrade pip
COPY . /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install vectorbtpro-1.11.1-py3-none-any.whl

USER root

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get purge -yqq make gcc g++ build-essential && \
    apt-get -yqq autoremove && \
    apt-get -yqq autoclean




RUN ls -la /app
RUN sleep 1
RUN chown -R r:r /app
ENV JUPYTER_CONFIG_DIR /app/config
RUN mkdir -p /usr/local/etc/jupyter/labconfig && chmod -R 777 /usr/local/etc/jupyter
RUN chmod -R 777 /app


USER r


ENV PATH="/home/r/.local/bin:${PATH}"

ENV DEBIAN_FRONTEND teletype

EXPOSE 4444
