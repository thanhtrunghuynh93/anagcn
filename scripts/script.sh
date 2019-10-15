PD=data/suhi_allmv_tmdb

python -u network_alignment.py \
--source_dataset ${PD}/allmv/graphsage/ \
--target_dataset ${PD}/tmdb/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
ANAGCN \
--embedding_dim 200 \
--emb_epochs 200 \
--lr 0.01 \
--num_GCN_blocks 2 \
--noise_level 0.001 \
--cuda \
--refinement_epoch 50 \
--refine \
--log 