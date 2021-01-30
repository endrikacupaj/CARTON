# Context Transformer with Stacked Pointer Networks for Conversational Question Answering over Knowledge Graphs

Neural semantic parsing approaches have been widely used for Question Answering (QA) systems over knowledge graphs. Such methods provide the flexibility to handle QA datasets with complex queries and a large number of entities. In this work, we propose a novel framework named CARTON, which performs multi-task semantic parsing for handling the problem of conversational question answering over a large-scale knowledge graph. Our framework consists of a stack of pointer networks as an extension of a context transformer model for parsing the input question and the dialog history. The framework generates a sequence of actions that can be executed on the knowledge graph. We evaluate CARTON on a standard dataset for complex sequential question answering on which CARTON outperforms all baselines. Specifically, we observe performance improvements in F1-score on eight out of ten question types compared to the previous state of the art. For logical reasoning questions, an improvement of 11 absolute points is reached.

## Requirements and Setup
Python version >= 3.7

PyTorch version >= 1.6.0

``` bash
# clone the repository
git clone https://github.com/endrikacupaj/CARTON.git
cd CARTON
pip install -r requirements.txt
```

## CSQA dataset
Our framework was evaluated on [CSQA](https://amritasaha1812.github.io/CSQA/) dataset. You can download the dataset from [here](https://amritasaha1812.github.io/CSQA/download/).

## Wikidata Knowlegde Graph
Since CSQA is based on Wikidata [Knowlegde Graph](https://www.wikidata.org/wiki/Wikidata:Main_Page), the authors provide a preproccesed version of it which can be used when working with the dataset.
You can download the preprocessed files from [here](https://zenodo.org/record/4052427#.YBU7xHdKjfZ).
After dowloading you will need to move them under the [knowledge_graph](knowledge_graph) directory.

## Annotate Dataset
Next, using the preproccesed Wikidata files we can annotate CSQA dataset with our proposed grammar.
``` bash
# annotate CSQA dataset with proposed grammar
python annotate_csqa/preprocess.py --partition train --annotation_task actions --read_folder /path/to/CSQA --write_folder /path/to/write
```

## Train Framework
For training you will need to adjust the paths in [args](args.py) file. At the same file you can also modify and experiment with different model settings.
``` bash
# train framework
python train.py
```

## Generate Actions
After the model has finished training we perform the inference in 2 steps.
First, we generate the actions and save them in JSON file using the trained model.
``` bash
# generate actions for a specific question type
python inference.py --question_type Clarification
```

## Execute Actions
Second, we execute the actions and get the results from Wikidata files.
``` bash
# execute actions for a specific question type
python action_executor/run.py --file_path /path/to/actions.json --question_type Clarification
```

## License
The repository is under [MIT License](LICENCE).

## Cite
Coming Soon!
