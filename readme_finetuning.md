# Fine-tuning

Firstly, let's see how to use Uni2TS to fine-tune a pre-trained model on your custom dataset.
Uni2TS uses the [Hugging Face datasets library](https://github.com/huggingface/datasets) to handle data loading, and we first need to convert your dataset into the Uni2TS format.
If your dataset is a simple pandas DataFrame, we can easily process your dataset with the following script.

1. To begin the process, add the path to the directory where you want to save the processed dataset into the ```.env``` file.
```shell
echo "CUSTOM_DATA_PATH=finetune_exp" >> .env
```

2. Run the following script to process the dataset into the required format. For the ```dataset_type``` option, we support `wide`, `long` and `wide_multivariate`.
```shell
python -m uni2ts.data.builder.simple AAPL src/data/sample_data/AAPL.csv --dataset_type wide
```

However, we may want validation set during fine-tuning to perform hyperparameter tuning or early stopping.
To additionally split the dataset into a train and validation split we can use the mutually exclusive ```date_offset``` (datetime string) or ```offset``` (integer) options which determines the last time step of the train set.
The validation set will be saved as DATASET_NAME_eval.
```shell
python -m uni2ts.data.builder.simple AAPL src/data/sample_data/AAPL.csv --date_offset '2019-12-31 23:00:00'
```

3. Finally, we can simply run the fine-tuning script with the appropriate [training](cli/conf/finetune/data/etth1.yaml) and [validation](cli/conf/finetune/val_data/etth1.yaml) data configuration files.
```shell
python -m cli.train \
  -cp conf/finetune \
  run_name=example_run \ 
  model=moirai_1.0_R_small \ 
  data=AAPl \ 
  val_data=AAPL
```