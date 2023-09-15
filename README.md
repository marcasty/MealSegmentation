# MealSegmentation

Here is our repo on segmenting meals into foods and masks which is then used for caloric estimation

Make sure to have docker and docker-compose installed on the machine. The models also only run with GPU NVIDIA support.

To build the docker image:

chmod +x bin/build.sh 
./bin/build.sh

To run the docker:

chmod +x bin/launch.sh 
./bin/launch.sh
