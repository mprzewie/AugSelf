declare -A datadirs
for DS in "cifar10" "cifar100" "pets" "flowers" "caltech101" "cars" "aircraft" "sun397";
do
  datadirs[$DS]="/shared/sets/datasets/vision"
done

datadirs["food101"]="/shared/sets/datasets/vision/food_101/"
datadirs["mit67"]="/shared/sets/datasets/vision/mit67_indoor_scenes/indoorCVPR_09/images_train_test/"
datadirs["stl10"]="/shared/sets/datasets/vision/stl10/"
datadirs["dtd"]="/home/przewiez/Downloads/dtd/"

declare -A metrics
for DS in ${!datadirs[@]};
do
  metrics[$DS]="top1"
done

for DS in "pets" "flowers" "caltech101" "aircraft"; do
  metrics[$DS]="class-avg"
done

