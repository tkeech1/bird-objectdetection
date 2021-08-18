install-deps:
	python3 -m pip install -r requirements.txt

start-jupyterlab:
	jupyter-lab

# object detection targets
get-base-model:
	wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz

# install tensorflow object detection
setup-tensorflow:
	git clone --depth 1 https://github.com/tensorflow/models | true
	cd models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install --use-feature=2020-resolver .

test-config:
	cd models/research && python object_detection/builders/model_builder_tf2_test.py

# xml_to_csv.py is from https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
xml-to-csv:
	python3 xml_to_csv.py -i data/annotated_images/train/ -o data/annotated_images/train/train.csv
	python3 xml_to_csv.py -i data/annotated_images/test/ -o data/annotated_images/test/test.csv

# generate_tfrecord.py is from https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
# may need to modify generate_tfrecord.py to correct imports since it references Python modules under models/
generate-tf-record:
	python3 generate_tfrecord.py --csv_input=data/annotated_images/train/train.csv --output_path=data/annotated_images/train/train.record --img_dir=data/annotated_images/train/
	python3 generate_tfrecord.py --csv_input=data/annotated_images/test/test.csv --output_path=data/annotated_images/test/test.record --img_dir=data/annotated_images/test/

train-model:
	python3 models/research/object_detection/model_main_tf2.py --pipeline_config_path=centernet_resnet50_v2_512x512_coco17_tpu-8_pipeline.config --alsologtostderr --model_dir=custom_models

#objdet-eval:
#	cd models/research/ && python3 /workspaces/deeplearn/models/research/object_detection/model_main_tf2.py --pipeline_config_path=/workspaces/deeplearn/pipeline_eval.config --alsologtostderr --model_dir=/workspaces/deeplearn/obj_det_model/eval
