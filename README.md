# Multi-Score Evaluation Metric
This is the code for Multi-Score evaluation metric proposed in the paper "[Generating Diverse Descriptions from Semantic Graphs](https://aclanthology.org/2021.inlg-1.1/)" by Jiuzhou Han, Daniel Beck and Trevor Cohn.

## Requirements

```
nltk==3.5
scipy==1.5.0
```

## Usage

To compute the Multi-Score of your test cases, run the `Multi-Score.py` script which receives as input the following parameters:

```
usage: Multi-Score.py [-h] -R REFERENCE -H1 HYPOTHESIS1 -H2 HYPOTHESIS2 -H3 HYPOTHESIS3 
                      [-nr NUM_REFS] [-nc NCORDER] [-nw NWORDER] [-b BETA]

arguments:
  -h, --help            show this help message and exit
  -R REFERENCE, --reference REFERENCE
                        reference translation
  -H1 HYPOTHESIS1, --hypothesis HYPOTHESIS
                        hypothesis translation
  -H2 HYPOTHESIS2, --hypothesis HYPOTHESIS
                        hypothesis translation
  -H3 HYPOTHESIS3, --hypothesis HYPOTHESIS
                        hypothesis translation
optional arguments:
  -nr NUM_REFS, --num_refs NUM_REFS
                        number of references (default=3)
  -nc NCORDER, --ncorder NCORDER
                        chrF metric: character n-gram order (default=6)
  -nw NWORDER, --nworder NWORDER
                        chrF metric: word n-gram order (default=2)
  -b BETA, --beta BETA  chrF metric: beta parameter (default=2)
```

An example on how to run to the evaluation script is available in `example.sh`. For instance, to obtain the Multi-Score (it will also give you BLEU score, ChrF++ score, Self-BLEU score), simply run the following command:

```
python3 Multi-Score.py -R data/references/test_reference -H1 data/predictions/prediction_01.txt -H2 data/predictions/prediction_02.txt -H3 data/predictions/prediction_03.txt
```

### Multiple References

In case of multiple references, they have to be stored in separated files and named reference0, reference1, reference2, etc.

### Citation
```
@inproceedings{han-etal-2021-generating,
    title = "Generating Diverse Descriptions from Semantic Graphs",
    author = "Han, Jiuzhou  and
      Beck, Daniel  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 14th International Conference on Natural Language Generation",
    month = aug,
    year = "2021",
    address = "Aberdeen, Scotland, UK",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.inlg-1.1",
    pages = "1--11"
}
```
