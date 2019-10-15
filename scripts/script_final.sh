PD=data/suhi_allmv_tmdb

python -u network_alignment.py \
--source_dataset ${PD}/allmv/graphsage/ \
--target_dataset ${PD}/tmdb/graphsage/ \
--groundtruth ${PD}/dictionaries/node,split=0.1.test.dict \
FINAL \
--train_dict ${PD}/dictionaries/node,split=0.1.train.dict