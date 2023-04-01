declare -A datadirs

main="/raid/NFS_SHARE/datasets/"

datadirs["imagenet100"]="${main}IN-100"

for DS in "cifar10" "cifar100" "pets" "flowers" "caltech101" "cars" "aircraft" "sun397";
do
  datadirs[$DS]=$main
done

datadirs["food101"]="${main}food_101/"
#datadirs["mit67"]="/storage/shared/datasets/mimit67_indoor_scenes/indoorCVPR_09/images_train_test/"
datadirs["stl10"]="${main}stl10/"

declare -A metrics
for DS in ${!datadirs[@]};
do
  metrics[$DS]="top1"
done

for DS in "pets" "flowers" "caltech101" "aircraft"; do
  metrics[$DS]="class-avg"
done

