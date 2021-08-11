name="tf_docker"

docker run -it -d --privileged --net=host     \
                --name $name                        \
                --shm-size="10g"                    \
                --env DISPLAY=$DISPLAY              \
                -v ~/.Xauthority:/root/.Xauthority  \
                -v /tmp/.X11-unix:/tmp/.X11-unix    \
                -v $PWD/root:/root/                 \
                ubuntu:18.04                        \
                bash

docker attach $name
