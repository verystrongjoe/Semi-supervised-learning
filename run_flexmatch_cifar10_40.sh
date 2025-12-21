# nohup ./run_flexmatch_cifar10_40.sh > flexmatch_cifar10_40.log 2>&1 &
python train.py --config config/classic_cv/flexmatch/flexmatch_cifar10_40_1111.yaml
python train.py --config config/classic_cv/flexmatch/flexmatch_cifar10_40_2222.yaml
python train.py --config config/classic_cv/flexmatch/flexmatch_cifar10_40_3333.yaml
python train.py --config config/classic_cv/flexmatch/flexmatch_cifar10_40_4444.yaml
python train.py --config config/classic_cv/flexmatch/flexmatch_cifar10_40_5555.yaml
