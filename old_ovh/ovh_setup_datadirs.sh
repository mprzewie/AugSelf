DATA_DIR="/storage/shared/datasets/"

declare -A datadirs
for DS in "cifar10" "cifar100" "pets" "flowers" "cars" "aircraft" "dtd" "sun397" "cub200";
do
  datadirs[$DS]=${DATA_DIR}
done

datadirs["caltech_101"]="${DATA_DIR}caltech_101/"
datadirs["food101"]="${DATA_DIR}food_101/"
datadirs["mit67"]="${DATA_DIR}mimit67_indoor_scenes/indoorCVPR_09/Images/"
datadirs["stl10"]="${DATA_DIR}stl10/"
datadirs["imagenet100"]="${DATA_DIR}ImageNet100_ssl/"
datadirs["cub200"]="/${DATA_DIR}CUB_200_2011/images/"

declare -A metrics
for DS in ${!datadirs[@]};
do
  metrics[$DS]="top1"
done

for DS in "pets" "flowers" "caltech_101" "aircraft"; do
  metrics[$DS]="class-avg"
done

