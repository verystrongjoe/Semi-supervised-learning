# nohup ./run_fixmatch_cifar10_40.sh > fixmatch_cifar10_40.log 2>&1 &
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_40_1111.yaml
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_40_2222.yaml
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_40_3333.yaml
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_40_4444.yaml
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_40_5555.yaml
