# build-hybrid-gpt



### Wikipedia dataset
Download **20220301.en** shard of [legacy-datasets/wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) dataset:
```bash
hf download legacy-datasets/wikipedia --repo-type dataset --include "data/20220301.en/*" --local-dir ./datasets/wikipedia_20220301_en
```

* Rows: 6_458_670

