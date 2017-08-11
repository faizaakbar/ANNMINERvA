MODELCODE=20170811
DAT=`date +%s`

singularity exec /data/perdue/singularity/simone/ubuntu16-cuda-ml.img python -m pdb mnv_run_st_epsilon.py \
  --compression gz \
  --data_dir /data/perdue/minerva/tensorflow/data  \
  --file_root minosmatch_nukecczdefs_genallzwitht_pcodecap66_127x50x25_xtxutuvtv_me1Amc_011  \
  --model_dir /data/perdue/minerva/tensorflow/models/11/${MODELCODE} \
  --log_name /data/perdue/minerva/tensorflow/logs/log_mnv_st_epsilon_11_${MODELCODE}_${DAT}.txt \
  --log_level DEBUG  \
  --do_training \
  --nodo_validaton \
  --nodo_testing \
  --nodo_prediction --pred_store_name /data/perdue/minerva/tensorflow/predictions//predictions_mnv_st_epsilon_11_${MODELCODE}
