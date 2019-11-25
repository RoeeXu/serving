image_name=yangyunlun/tensorflow-serving-gpu
docker stop $(docker ps | grep $image_name | awk '{print $1}') 
docker rm $(docker ps | grep $image_name | awk '{print $1}') 


USER_NAME=yangyunlun
qa_model_path=/search/data3/hzsun/tf-serving-model-bak/qacbert-xla-03-ckpt40-layer1+7-dim384.8/
mrc_model_path=/search/data3/hzsun/tf-serving-model-bak/span-mrcbert-02-ckpt200-layer1+7-dim384.6/
custom_op_path=/search/data3/hzsun/cuda_codes/serving_10.0/
mount_op_path=/custom_op
custom_ops=${mount_op_path}/kernel_fo_conv.so:${mount_op_path}/kernel_fo_pooling.so

for i in 0 1 2 3 4 5 6 7
do 
    gpu_id=${i}
    qa_port=$[${i}+8500]
    mrc_port=$[${i}+8600]

    docker run \
      --runtime=nvidia \
      -e CUDA_VISIBLE_DEVICES=${i} \
      -p $[${qa_port}-1000]:8500 \
      -p ${qa_port}:8501 \
      --mount type=bind,source=$qa_model_path,target=/models/qa_model \
      --mount type=bind,source=$custom_op_path,target=$mount_op_path \
      -dit $USER_NAME/tensorflow-serving-gpu \
      --port=8500 --rest_api_port=8501 \
      --custom_ops=$custom_ops \
      --model_name=qa --model_base_path=/models/qa_model \
      --per_process_gpu_memory_fraction=0.5 \
      --tensorflow_session_parallelism=5

    docker run \
      --runtime=nvidia \
      -e CUDA_VISIBLE_DEVICES=${i} \
      -p $[${mrc_port}-1000]:8500 \
      -p ${mrc_port}:8501 \
      --mount type=bind,source=$mrc_model_path,target=/models/mrc_model \
      --mount type=bind,source=$custom_op_path,target=$mount_op_path \
      -dit $USER_NAME/tensorflow-serving-gpu \
      --port=8500 --rest_api_port=8501 \
      --custom_ops=$custom_ops \
      --model_name=mrc --model_base_path=/models/mrc_model \
      --per_process_gpu_memory_fraction=0.5 \
      --tensorflow_session_parallelism=5
done
