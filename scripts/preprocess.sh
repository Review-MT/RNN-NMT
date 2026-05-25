export CUDA_VISIBLE_DEVICES=1
src=$1
tgt=$2
lang=${2}-${1}

export DATA_PREP="/mnt/storage/divya/exam"    # $DATA/$lang
export DATA_HOME=$DATA_PREP/rnndata

python ../preprocess.py \
  -train_src $DATA_PREP/clean.train.$src \
  -train_tgt $DATA_PREP/clean.train.$tgt \
  -train_xe_src $DATA_PREP/clean.train.$src \
  -train_xe_tgt $DATA_PREP/clean.train.$tgt \
  -train_pg_src $DATA_PREP/clean.train.$src \
  -train_pg_tgt $DATA_PREP/clean.train.$tgt \
  -valid_src $DATA_PREP/valid.$src \
  -valid_tgt $DATA_PREP/valid.$tgt \
  -test_src $DATA_PREP/test.$src \
  -test_tgt $DATA_PREP/test.$tgt \
  -save_data $DATA_HOME/bprocessed_all \
  -embedding_type bert
