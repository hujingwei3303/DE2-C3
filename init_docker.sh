#!/bin/bash

apt-get update -y

apt-get upgrade -y

apt-get install -y apt-transport-https ca-certificates curl software-properties-common

echo "adding docker repo"

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
apt-get update -y
apt-get install -y docker-ce
echo "adding docker-compose"
curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-Linux-x86_64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
