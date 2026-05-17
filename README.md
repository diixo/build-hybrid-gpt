# build-hybrid-gpt



### Wikipedia dataset
Download **20220301.en** shard of [legacy-datasets/wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) dataset:
```bash
hf download legacy-datasets/wikipedia --repo-type dataset --include "data/20220301.en/*" --local-dir ./wikipedia-20220301-en
```

* Rows: 6_458_670

