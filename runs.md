- [x] Baselines
### single sentiment task
```bash
python trainer.py --batch_size 32 --ro -1 --task sentiment --type 1
```
* log path: sentiment-type:1-ro:-1
* acc on dev(pytorch): 69.54, reported acc on dev(dynet): 67.4

#### race leakage on sentiment task
```bash
python attacker.py --ro -1 --task race --model sentiment-type:1-ro:-1/model_best.pth.tar --init 1 --batch_size 32
```
* log path: attacker-sentiment-type:1-ro:-1
* acc on dev(pytorch): 65.33, reported acc on dev(dynet): 64.5

### single race task
```bash
python trainer.py --batch_size 32 --ro -1 --task race --type 2 --lr 0.01 
```
* log path: race-type:2-ro:-1
* acc on dev(pytorch): 82.4, reported acc on dev(dynet): 83.9

### single unbalanced sentiment task
```bash
python trainer.py --batch_size 32 --ro -1 --task unbalanced_race --type 1
```
* log path: unbalanced_race-type:1-ro:-1
* acc on dev(pytorch): 80.19, reported acc on dev(dynet): 79.5

### single unbalanced race task
```bash
python trainer.py --batch_size 32 --ro -1 --task unbalanced_race --type 2
```
* log path: unbalanced_race-type:2-ro:-1
* acc on dev(pytorch): 86.44, reported acc on dev(dynet): --

#### race leakage on sentiment task, unbalanced
```bash
python attacker.py --batch_size 32 --ro -1 --task unbalanced_race --model unbalanced_race-type:1-ro:-1/model_best.pth.tar
```
* log path: attacker-unbalanced_race-type:1-ro:-1
* acc on dev(pytorch): 81.21, reported acc on dev(dynet): 73.5

#### Adv, race leakage on sentiment task
```bash
python trainer.py --batch_size 32 --ro 1 --task sent_race --num_adv 1 --lr 0.01
```
* log path: 
* acc on sentiment dev(pytorch): , reported sentiment acc on dev(dynet): 64.7
* acc on race dev(pytorch): , reported race acc on dev(dynet): 56.0

```
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task race --model sentiment-type:1-ro:-1/best_model --init 1

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task race --type 2

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task gender --type 2

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_gender --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --task gender --model sent_gender-type:1-ro:-1/best_model --init 1


python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --model mention_race-type:1-ro:-1/best_model --init 1


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention2_gender --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention2_gender --model mention2_gender-type:1-ro:-1/best_model --init 1

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention2_gender --type 2

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention_age --type 2

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro -1 --task mention_age --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_age --model mention_age-type:1-ro:-1/best_model --init 1
```

- [x] Unbalanced Sentiment
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task unbalanced_race --type 2

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task unbalanced_race --type 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task unbalanced_race --model unbalanced_race-type:1-ro:-1/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-type:1-ro:-1/best_model --init 1

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:1.0/best_model --init 1


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 500
?python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50 --adv_size 500

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:1000/best_model --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50 --adv_size 1000

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50 --adv_size 2000

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50 --adv_size 5000

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 1 --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --model unbalanced_race-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50 --adv_size 8000


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2 --model unbalanced_race-n_adv:2-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3 --model unbalanced_race-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task unbalanced_race --num_adv 5
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5 --model unbalanced_race-n_adv:5-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 0.5 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:0.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.5 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:1.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 2.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:2.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:3.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task unbalanced_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task sent_race --num_adv 1 --model unbalanced_race-n_adv:1-ro:5.0/epoch_50

``` 

- [x] Adversarials
* Race Adversarial
```bash
python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_10
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_50 --init 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_60
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unseen_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task unseen_race --num_adv 1 --model sent_race-n_adv:1-ro:1.0/epoch_50 --init 1

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2 --model sent_race-n_adv:2-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 2 --model sent_race-n_adv:2-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3 --model sent_race-n_adv:3-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 3 --model sent_race-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 123 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5
python attacker.py --dynet-seed 123 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5 --model sent_race-n_adv:5-ro:1.0/best_model
python attacker.py --dynet-seed 123 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 5 --model sent_race-n_adv:5-ro:1.0/epoch_50

```

* Other Adversarials
```bash

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_race --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_race --num_adv 1 --model mention_race-n_adv:1-ro:1.0/best_model
```

- [x] Age Adversarials
```bash
python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --model mention_age-n_adv:2-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --model mention_age-n_adv:2-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 3 --model mention_age-n_adv:3-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 3 --model mention_age-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 5
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 5 --model mention_age-n_adv:5-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 5 --model mention_age-n_adv:5-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 500
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 500 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 500 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:500/epoch_100

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 1000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 2000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 2000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_100

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 5000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 8000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 15000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 15000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:15000/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 1 --adv_size 15000 --model mention_age-n_adv:1-ro:1.0-adv_hid_size:15000/epoch_100


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention_age --adv_size 8000 --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --adv_size 8000 --model mention_age-n_adv:2-ro:1.0-adv_hid_size:8000/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention_age --num_adv 2 --adv_size 8000 --model mention_age-n_adv:2-ro:1.0-adv_hid_size:8000/best_model


python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 0.5 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:0.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:0.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.5 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:1.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 2.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:2.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:2.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:3.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:3.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:5.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:5.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention_age --num_adv 1 --model mention_age-n_adv:1-ro:5.0/epoch_60
```

- [x] Gender2 Adversarials
```bash
python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2 --model mention2_gender-n_adv:2-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2 --model mention2_gender-n_adv:2-ro:1.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 2 --model mention2_gender-n_adv:2-ro:1.0/epoch_60

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 3
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 3 --model mention2_gender-n_adv:3-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 3 --model mention2_gender-n_adv:3-ro:1.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 5
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 5 --model mention2_gender-n_adv:5-ro:1.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 5 --model mention2_gender-n_adv:5-ro:1.0/epoch_50


python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 500
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 500 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 500 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:500/epoch_60

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 1000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 2000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 5000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task mention2_gender --num_adv 1 --adv_size 8000 --model mention2_gender-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 0.5 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:0.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:0.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.5 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.5/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:1.5/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:2.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:2.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:2.0/epoch_60

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:3.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:3.0/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task mention2_gender --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:5.0/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task mention2_gender --num_adv 1 --model mention2_gender-n_adv:1-ro:5.0/epoch_50
```

- [x] lstm parameters reduction - Sentiment-Race
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 200
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 200 --model sent_race-type:1-hid:200-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 100
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 100 --model sent_race-type:1-hid:100-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 50 --model sent_race-type:1-hid:50-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 10
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 10 --model sent_race-type:1-hid:10-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 2 --model sent_race-type:1-hid:2-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task sent_race --enc_size 1 --model sent_race-type:1-hid:1-ro:-1/best_model
```

- [x] lstm parameters reduction - Mention-Race
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 200
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 200 --model mention_race-type:1-hid:200-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 100
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 100 --model mention_race-type:1-hid:100-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 50 --model mention_race-type:1-hid:50-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 10
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 10 --model mention_race-type:1-hid:10-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 2
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 2 --model mention_race-type:1-hid:2-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task mention_race --enc_size 1 --model mention_race-type:1-hid:1-ro:-1/best_model
```

- [x] Bigger Adversarial
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --adv_size 500
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 500 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:500/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 500 --att_hid_size 500 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:500/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 500 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:500/epoch_50

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --adv_size 1000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 1000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:1000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 1000 --att_hid_size 1000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:1000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 1000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:1000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:2000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000 --att_hid_size 2000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:2000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 2000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:2000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:3000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000 --att_hid_size 3000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:3000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 3000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:3000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:5000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000 --att_hid_size 5000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:5000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 5000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:5000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:6000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000 --att_hid_size 6000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:6000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 6000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:6000/epoch_50

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:8000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000 --att_hid_size 8000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:8000/best_model
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task sent_race --num_adv 1 --adv_size 8000 --model sent_race-n_adv:1-ro:1.0-adv_hid_size:8000/epoch_50
```

- [x] Lambda Change
```bash
python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race --model sent_race-n_adv:1-ro:0.5/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 0.5 --task sent_race --model sent_race-n_adv:1-ro:0.5/best_model

python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race --model sent_race-n_adv:1-ro:1.5/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.5 --task sent_race --model sent_race-n_adv:1-ro:1.5/best_model


python trainer.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race --model sent_race-n_adv:1-ro:2.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 2.0 --task sent_race --model sent_race-n_adv:1-ro:2.0/best_model

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 3.0 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --model sent_race-n_adv:1-ro:3.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --model sent_race-n_adv:1-ro:3.0/epoch_60
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 3.0 --task sent_race --model sent_race-n_adv:1-ro:3.0/best_model

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 5.0 --task sent_race
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task sent_race --model sent_race-n_adv:1-ro:5.0/epoch_50
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 5.0 --task sent_race --model sent_race-n_adv:1-ro:5.0/best_model
```

- [x] Race-Sentiment
```bash
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro -1 --task race_sent --model race-type:2-ro:-1/best_model

python trainer.py --dynet-seed 12345 --dynet-gpus 1 --dynet-autobatch 1 --ro 1.0 --task race_sent --num_adv 1
python attacker.py --dynet-seed 12345 --dynet-autobatch 1 --ro 1.0 --task race_sent --num_adv 1 --model race_sent-n_adv:1-ro:1.0/best_model
```
