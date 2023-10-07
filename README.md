# MealSegmentation

Here is our repo on segmenting meals into foods and masks which is then used for caloric estimation

Make sure to have docker and docker-compose installed on the machine. The models also only run with GPU NVIDIA support.

To build the docker image:

chmod +x bin/build.sh 
./bin/build.sh

To run the docker:

chmod +x bin/launch.sh 
./bin/launch.sh

To enter the docker:

docker exec -it meal-segmentation:latest bash

Once inside the docker:

cd src/
chmod +x bin/grounding_dino.sh
./bin/grounding_dino.sh