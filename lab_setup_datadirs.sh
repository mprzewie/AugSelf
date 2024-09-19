DATA_DIR="/home/marcin.przewiezlikowki/datasets/"

declare -A datadirs
for DS in "cifar10" "cifar100" "pets" "flowers" "flowers-5shot" "flowers-10shot" "caltech101" "cars" "aircraft" "sun397" "dtd" "celeba";
do
  datadirs[$DS]=${DATA_DIR}
done

datadirs["food101"]="${DATA_DIR}food_101/"
datadirs["mit67"]="${DATA_DIR}mit67_indoor_scenes/indoorCVPR_09/images_train_test/"
datadirs["stl10"]="${DATA_DIR}stl10/"
datadirs["imagenet100"]="${DATA_DIR}IN-100/"
datadirs["cub200"]="${DATA_DIR}/CUB_200_2011/images_train_test"


declare -A metrics
for DS in ${!datadirs[@]};
do
  metrics[$DS]="top1"
done

for DS in "pets" "flowers" "caltech101" "aircraft"; do
  metrics[$DS]="class-avg"
done

