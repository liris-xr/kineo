# Evaluation

We evaluated Kineo on the EgoHumans and Human3.6M datasets. To reproduce our results, first download and preprocess the datasets using the provided scripts:
```sh
python scripts/download_h36m_dataset.py <path-to-h36m-dataset>
python scripts/preprocess_h36m_dataset.py <path-to-h36m-dataset>

python scripts/download_egohumans_dataset.py <path-to-egohumans-dataset>
python scripts/preprocess_egohumans_dataset.py <path-to-egohumans-dataset>
```

Once the datasets are prepared, you can run evaluation using the corresponding evaluation scripts:
```sh
python experiments/h36m_eval.py <path-to-h36m-dataset> configs/experiments/benchmarks/h36m_benchmark_nlf_estRt_estK_estD.yaml

python experiments/egohumans_eval.py <path-to-egohumans-dataset> configs/experiments/benchmarks/egohumans_benchmark_nlf_estRt_estK_estD.yaml
```
All configurations used in the paper are available in the `configs` directory.