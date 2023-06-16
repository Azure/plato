FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

RUN apt-get update \
	# Otherwise installing default-jre-headless fails
	&& mkdir -p /usr/share/man/man1/ \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y default-jre-headless git memcached \
	&& apt clean \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src
COPY requirements.txt /opt/src
RUN pip install -Ur requirements.txt
