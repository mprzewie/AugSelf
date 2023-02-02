DATA_DIR="/storage/shared/datasets/"

declare -A datadirs
for DS in "cifar10" "cifar100" "pets" "flowers" "caltech101" "cars" "aircraft" "sun397" "dtd";
do
  datadirs[$DS]=${DATA_DIR}
done

datadirs["food101"]="${DATA_DIR}food_101/"
datadirs["mit67"]="${DATA_DIR}mimit67_indoor_scenes/indoorCVPR_09/images_train_test/"
datadirs["stl10"]="${DATA_DIR}stl10/"
datadirs["imagenet100"]="${DATA_DIR}ImageNet100_ssl/"

declare -A metrics
for DS in ${!datadirs[@]};
do
  metrics[$DS]="top1"
done

for DS in "pets" "flowers" "caltech101" "aircraft"; do
  metrics[$DS]="class-avg"
done

