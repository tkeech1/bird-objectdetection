## Object detection using transfer learning on a custom data set

1) Install dependencies

    `make install-deps`

2) Download and extract the base model to use for transfer learning

    ```
    make get-base-model
    tar xzf centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz
    ```

3) Download, install and setup Tensorflow object detection

    `make setup-tensorflow`

4) Test the Tensorflow configuration

    `make test-config`

5) Create a set of images labeled with bounding boxes for fine-tuning the base model. Use the labelImg Python package to create bounding boxes for each image (bounding box details are saved as .xml files). Ensure images are sized properly for the model (i.e. 512x512, etc.) prior to creating the bounding boxes. `youtube_vide_extract.ipynb` shows an example of how to extract frames from a YouTube video and store them as a 512x512 JPEG. 

6) Run the Python script to convert the bounding box xml files to CSV format.
    `make xml-to-csv`

7) Generate TF records from the CSV files
    `make generate-tf-record`

8) Create a `label_map.pbtext` file that provides a mapping from IDs (you can make these up) to class names. 

9) Edit the pipeline configuration file.
    ```
    num_classes: Set to the number of prediction classes in the custom model. This should be the same as the number of classes in the label_map.pbtext file. 
    fine_tune_checkpoint: Set to the path to the base model checkpoint - make sure to include ckpt-# at the end of the path
    fine_tune_checkpoint_type: Set to "detection"
    train_input_reader.label_map_path: Path to the label_map.pbtext file
    train_input_reader.tf_record_input_reader.input_path: Path to the train.record TF record file
    eval_input_reader.label_map_path: Path to the label_map.pbtext file
    eval_input_reader.tf_record_input_reader.input_path: Path to the test.record TF record file
    ```
10) Train
    `make train-model`


Issues:
* If you experience GPU OOM issues, clear/shutdown any kernels running in Jupyter Lab


Refs:
* https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md
* https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html#
* https://www.tensorflow.org/tutorials/load_data/tfrecord
* https://www.tensorflow.org/hub/tutorials/tf2_object_detection
