#!/bin/bash

apt-get update -y

apt-get upgrade -y

apt-get install -y apt-transport-https ca-certificates curl software-properties-common python3-pip python3-dev build-essential

echo "adding docker repo"

git clone https://github.com/hujingwei3303/DE2-C3.git

pip3 install -r DE2-C3/requirement.txt




