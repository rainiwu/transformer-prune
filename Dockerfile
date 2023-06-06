FROM python:3.11-3-slim

# create a user inside the container
ARG USER=prune
ARG USERDIR=/home/${USER}
RUN adduser --disabled-password --gecos "" ${USER}

# install dependency manager
RUN pip install -U poetry

# copy current directory
RUN mkdir ${USERDIR}/prune
COPY --chown=${USER} . ${USERDIR}/prune

# build result
RUN cd ${USERDIR}/prune && poetry build && cd dist && pip install prune-*.tar.gz

USER ${USER}
ENTRYPOINT []
