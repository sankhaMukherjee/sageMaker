# Batch Inference

For batch inference you need to follow the following steps:

1. Create a folder containing your model in the folder structure as shown below
2. This entire folder needs to be converted into a `.tar.gz` file.
3. Upload the file into a particular S3 bucket
4. Use the Python API to run the inference

```
├── code
│   ├── inference.py
│   └── requirements.txt
└── 1 <--------------------- That is a model number. You can use any number you please 
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

There is a file that allows you to generate this folder structure, zip it, upload the zipped file into s3, and clean up the temp folder
it just created. This way any change that you make will be automatically be first uploaded into the s3 bucket in the right folder, and
then, you can use the script that uses the Python API to run the command.

For doing this, 

- download one of the pretrained models into a folder (call this `/path/to/pretrainedFolder/<model number>`). 
- In the [`src/part_07_batchInference/config.json`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/config.json) file, update the `"modelFolder"` to point to this folder. 
- The code [`src/part_07_batchInference/utils/createFolderStructure.py`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/utils/createFolderStructure.py) will help you create the folder structure, zip it into a `.tar.gz` file, and upload it into your s3 bucket (defined by `"s3bucket"` in 
- [`src/part_07_batchInference/config.json`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/config.json)) and upload it in the S3 file defined in `"s3model"` in [`src/part_07_batchInference/config.json`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/config.json).
- Make sure that theinference code is present in the folder pointed to by `"codeFolder"` in [`src/part_07_batchInference/config.json`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/config.json).
- Once this is done, run the script [`src/part_07_batchInference/utils/createFolderStructure.py`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/utils/createFolderStructure.py).
- After this, the script [src/part_07_batchInference/batchInference.py](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/batchInference.py) can be used for doing batch inference. 

It is possible to use a Make command for doing both things together. Use the command:

`make batchInference` to run both scripts.

# Get information about the model ...

For the saved model, first understand the parameters for the input and the output of the model. This can be
done easily by using the `saved_model_cli` command. This is going to show you what the model will expect in
terms of inputs and outputs ...

```bash
saved_model_cli show --dir ../temp/model/1 --all 
```

The return from this command is shown below. This will show what is required for the model as an input and
output. In this case, for the `'serving_default'` signature, we see that the model needs inputs in the form
(-1, 28, 28, 0), and outputs results in the form (-1, 10), which happens to be a softmax output. 

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['conv2d_input'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 28, 28, 1)
        name: serving_default_conv2d_input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['dense_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 10)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          conv2d_input: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='conv2d_input')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #2
    <many more options ...>
```

# References

1. [Performing batch inference with TensorFlow Serving in Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/performing-batch-inference-with-tensorflow-serving-in-amazon-sagemaker/)
2. [Official TensorFlow Serving Tutorial](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)
3. [TensorFlow Serving (YouTube Video)](https://www.youtube.com/watch?v=zpKm8OxDBwE)
4. [Content Types](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-inference.html)


