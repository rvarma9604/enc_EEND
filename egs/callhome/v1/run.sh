#!/bin/bash

stage=6

# 1 first phase training
# 2 average first phase
# 3 infer first phase
# 4 score first phase
# 5 first phase results
# 6 second phase train
# 7 average second phase
# 8 infer second phase
# 9 score second phase
# 10 adapt second phase
# 11 adapt average
# 12 adapt infer
# 13 adapt scoring
# 14 adapt results 



# kaldi style data directory
# audio files should be reguar wav files  (should not be piped)
initial_train_set=data/simu/data/swb_sre_tr_ns2_beta2_100000
initial_valid_set=data/simu/data/swb_sre_cv_ns2_beta2_500

final_train_set=/data1/rajatv/data/simu/data/combined_tr
final_valid_set=/data1/rajatv/data/simu/data/combined_cv
#final_train_set=data/simu/data/swb_sre_tr_ns4_beta2_100000
#final_valid_set=data/simu/data/swb_sre_cv_ns4_beta2_500

adapt_set=data/eval/callhome1_spk2
adapt_valid_set=data/eval/callhome2_spk2


# Base config files for {train,infer}.py
train_config=conf/train.yaml
train_25_config=conf/train_25.yaml
infer_config=conf/infer.yaml
adapt_config=conf/adapt.yaml

# Additional arguments passed to {train,infer}.py
# You need to edit the base config files above
train_args=
train_25_args=
infer_args=
adapt_args=

# Initial Phase Model averaging options
initial_average_start=91
initial_average_end=100

# Final Phase Model averaging options
final_average_start=21
final_average_end=25

# Adapted model averaging otions
adapt_average_start=91
adapt_average_end=100


# Resume training from snapshot at this epoch
# TODO: not tested
resume=-1

# Debug purpose
debug=


. path.sh
. cmd.sh
. parse_options.sh || exit

set -eu

# ignoring Debug pipeline


# Parse the config file to set bash variables like: $train_frame_shift, $infer_gpu
eval `yaml2bash.py --prefix train $train_config`
eval `yaml2bash.py --prefix train2 $train_25_config`
eval `yaml2bash.py --prefix infer $infer_config`

# Append gpu reservation flag to the queuing command
# if [ $train_gpu -le 0 ]; then
#     train_cmd+=" --gpu 1"
# fi
# if [ $train2_gpu -le 0 ]; then
# 	train2_cmd+=" --gpu 1"
# fi
# if [ $infer_gpu -le 0 ]; then
#     infer_cmd+=" --gpu 1"
# fi


# Build directory names for the experiment
#  - Initial Phase Training
#     exp/diarize/model/{initial_train_id}.{initial_valid_id}.{train_config_id}
#  - Final Phase Training
#     exp/diarize/model/{final_train_id}.{final_valid_id}.{train_25_config_id}
#
#  - Decoding
#     exp/diarize/infer/{initial_train_id}.{initial_valid_id}.{train_config_id}.{infer_config_id}
#     exp/diarize/infer/{final_train_id}.{final_valid_id}.{train_25_config_id}.{infer_config_id}
#  - Scoring
#     exp/diarize/scoring/{initial_train_id}.{final_valid_id}.{train_config_id}.{infer_config_id}
#     exp/diarize/scoring/{final_train_id}.{final_valid_id}.{train_25_config_id}.{infer_config_id}
#  - Adaptation from non-adapted averaged model
#     exp/diarize/model/{initial_train_id}.{initial_valid_id}.{train_config_id}.{avgid}.{adapt_config_id}
#     exp/diarize/model/{final_train_id}.{final_valid_id}.{train_25_config_id}.{avgid}.{adapt_config_id}


initial_train_id=$(basename $initial_train_set)
initial_valid_id=$(basename $initial_valid_set)

final_train_id=$(basename $final_train_set)
final_valid_id=$(basename $final_valid_set)

train_config_id=$(echo $train_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
train_25_config_id=$(echo $train_25_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer_config_id=$(echo $infer_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
adapt_config_id=$(echo $adapt_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')


# Additional arguments are added to config_id
train_config_id+=$(echo $train_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
train_25_config_id+=$(echo $train_25_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
infer_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
adapt_config_id+=$(echo $adapt_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')



# First Phase of Training
initial_model_id=$initial_train_id.$initial_valid_id.$train_config_id
initial_model_dir=exp/diarize/model/$initial_model_id
## echo $initial_model_id
## echo $initial_model_dir
if [ $stage -le 1 ]; then
    echo "training model at $initial_model_dir."
    if [ -d $initial_model_dir ]; then
        echo "$initial_model_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$initial_model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $train_config \
            $train_args \
            $initial_train_set $initial_valid_set $initial_model_dir \
            || exit 1
fi
#exit


# Averaging after First Phase
initial_ave_id=avg${initial_average_start}-${initial_average_end}
## echo $initial_ave_id
if [ $stage -le 2 ]; then
    echo "averaging model parameters into $initial_model_dir/$initial_ave_id.nnet.npz"
    if [ -s $initial_model_dir/$initial_ave_id.nnet.npz ]; then
        echo "$initial_model_dir/$initial_ave_id.nnet.npz already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    models=`eval echo $initial_model_dir/snapshot_epoch-{$initial_average_start..$initial_average_end}`
    model_averaging.py $initial_model_dir/$initial_ave_id.nnet.npz $models || exit 1
fi



# Inference First Phase
initial_infer_dir=exp/diarize/infer/$initial_model_id.$initial_ave_id.$infer_config_id
## echo $initial_infer_dir
if [ $stage -le 3 ]; then
    echo "inference at $initial_infer_dir"
    if [ -d $initial_infer_dir ]; then
        echo "$initial_infer_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$initial_infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            infer.py \
            -c $infer_config \
            $infer_args \
            data/eval/$dset \
            $initial_model_dir/$initial_ave_id.nnet.npz \
            $initial_infer_dir/$dset \
            || exit 1
    done
fi


# Scoring First Phase
initial_scoring_dir=exp/diarize/scoring/$initial_model_id.$initial_ave_id.$infer_config_id
## echo $initial_scoring_dir
if [ $stage -le 4 ]; then
    echo "scoring at $initial_scoring_dir"
    if [ -d $initial_scoring_dir ]; then
        echo "$initial_scoring_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$initial_scoring_dir/$dset/.work
        mkdir -p $work
        find $initial_infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        make_rttm.py --median=$med --threshold=$th \
            --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
            $work/file_list_$dset $initial_scoring_dir/$dset/hyp_${th}_$med.rttm
        md-eval.pl -c 0.25 \
            -r data/eval/$dset/rttm \
            -s $initial_scoring_dir/$dset/hyp_${th}_$med.rttm > $initial_scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
    done
fi


# First Phase Results
if [ $stage -le 5 ]; then
    for dset in callhome2_spk2; do
        best_score.sh $initial_scoring_dir/$dset
    done
fi
echo "Finished First Phase!"



# Second Phase Training
final_model_id=$final_train_id.$final_valid_id.$train_25_config_id
final_model_dir=exp/diarize/model/$final_model_id
## echo $final_model_id
## echo $final_model_dir

echo $train_cmd
echo $train2_cmd
echo $train_args
echo $train_25_args

if [ $stage -le 6 ]; then
    echo "training model at $final_model_dir."
    if [ -d $final_model_dir ]; then
        echo "$final_model_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$final_model_dir/.work
    mkdir -p $work
    $train2_cmd $work/train.log \
        train.py \
            -c $train_25_config \
            $train_25_args \
	        --initmodel /data1/rajatv/enc_EEND/egs/callhome/v2/exp/diarize/model/swb_sre_tr_ns2_beta2_100000.swb_sre_cv_ns2_beta2_500.train/avg91-100.nnet.npz \
            $final_train_set $final_valid_set $final_model_dir \
            || exit 1
fi
exit

# Averaging after Second Phase
final_ave_id=avg${final_average_start}-${final_average_end}
echo $final_ave_id

if [ $stage -le 7 ]; then
    echo "averaging model parameters into $final_model_dir/$final_ave_id.nnet.npz"
    if [ -s $final_model_dir/$final_ave_id.nnet.npz ]; then
        echo "$final_model_dir/$final_ave_id.nnet.npz already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    models=`eval echo $final_model_dir/snapshot_epoch-{$final_average_start..$final_average_end}`
    model_averaging.py $final_model_dir/$final_ave_id.nnet.npz $models || exit 1
fi


# Inference Second Phase
final_infer_dir=exp/diarize/infer/$final_model_id.$final_ave_id.$infer_config_id
if [ $stage -le 8 ]; then
    echo "inference at $final_infer_dir"
    if [ -d $final_infer_dir ]; then
        echo "$final_infer_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$final_infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            infer.py \
            -c $infer_config \
            $infer_args \
            data/eval/$dset \
            $final_model_dir/$final_ave_id.nnet.npz \
            $final_infer_dir/$dset \
            || exit 1
    done
fi


# Scoring Second Phase
final_scoring_dir=exp/diarize/scoring/$final_model_id.$final_ave_id.$infer_config_id
if [ $stage -le 9 ]; then
    echo "scoring at $final_scoring_dir"
    if [ -d $final_scoring_dir ]; then
        echo "$final_scoring_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$final_scoring_dir/$dset/.work
        mkdir -p $work
        find $final_infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        make_rttm.py --median=$med --threshold=$th \
            --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
            $work/file_list_$dset $final_scoring_dir/$dset/hyp_${th}_$med.rttm
        md-eval.pl -c 0.25 \
            -r data/eval/$dset/rttm \
            -s $final_scoring_dir/$dset/hyp_${th}_$med.rttm > $final_scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
    done
fi


# Adapting model after Second Phase
adapt_model_dir=exp/diarize/model/$final_model_id.$final_ave_id.$adapt_config_id
if [ $stage -le 10 ]; then
    echo "adapting model at $adapt_model_dir"
    if [ -d $adapt_model_dir ]; then
        echo "$adapt_model_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$adapt_model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $adapt_config \
            $adapt_args \
            --initmodel $final_model_dir/$final_ave_id.nnet.npz \
            $adapt_set $adapt_valid_set $adapt_model_dir \
                || exit 1
fi


# Adapt averaging
adapt_ave_id=avg${adapt_average_start}=${adapt_average_end}
if [ $stage -le 11 ]; then
    echo "averaging models into $adapt_model_dir/$adapt_ave_id.nnet.gz"
    if [ -s $adapt_model_dir/$adapt_ave_id.nnet.npz ]; then
        echo "$adapt_model_dir/$adapt_ave_id.nnet.npz already exists."
        echo " if you want to retry, please remove it."
        exit 1
    fi
    models=`eval echo $adapt_model_dir/snapshot_epoch-{$adapt_average_start..$adapt_average_end}`
    model_averaging.py $adapt_model_dir/$adapt_ave_id.nnet.npz $models || exit 1
fi 


# Adapt inference
adapt_infer_dir=exp/diarize/infer/$final_model_id.$final_ave_id.$adapt_config_id.$adapt_ave_id.$infer_config_id
if [ $stage -le 12 ]; then
    echo "inference at $adapt_infer_dir"
    if [ -d $adapt_infer_dir ]; then
        echo "$adapt_infer_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$adapt_infer_dir/$dset/.work
        mkdir -p $work
        $train_cmd $work/infer.log \
            infer.py -c $infer_config \
            data/eval/${dset} \
            $adapt_model_dir/$adapt_ave_id.nnet.npz \
            $adapt_infer_dir/$dset \
            || exit 1
    done
fi


# Adapt Scoring
adapt_scoring_dir=exp/diarize/scoring/$final_model_id.$final_ave_id.$adapt_config_id.$adapt_ave_id.$infer_config_id
if [ $stage -le 13 ]; then
    echo "scoring at $adapt_scoring_dir"
    if [ -d $adapt_scoring_dir ]; then
        echo "$adapt_scoring_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$adapt_scoring_dir/$dset/.work
        mkdir -p $work
        find $adapt_infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        make_rttm.py --median=$med --threshold=$th \
            --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
            $work/file_list_$dset $adapt_scoring_dir/$dset/hyp_${th}_$med.rttm
        md-eval.pl -c 0.25 \
            -r data/eval/$dset/rttm \
            -s $adapt_scoring_dir/$dset/hyp_${th}_$med.rttm > $adapt_scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
    done
fi


# Final scoring
if [ $stage -le 14 ]; then
    for dset in callhome2_spk2; do
        best_score.sh $adapt_scoring_dir/$dset
    done
fi
echo "Finished !"
