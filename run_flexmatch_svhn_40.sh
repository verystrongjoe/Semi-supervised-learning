# nohup ./run_flexmatch_svhn_40.sh > flexmatch_svhn_40.log 2>&1 &
python train.py --c config/classic_cv/flexmatch/flexmatch_svhn_40_1111.yaml
python train.py --c config/classic_cv/flexmatch/flexmatch_svhn_40_2222.yaml
python train.py --c config/classic_cv/flexmatch/flexmatch_svhn_40_3333.yaml
python train.py --c config/classic_cv/flexmatch/flexmatch_svhn_40_4444.yaml
python train.py --c config/classic_cv/flexmatch/flexmatch_svhn_40_5555.yaml
