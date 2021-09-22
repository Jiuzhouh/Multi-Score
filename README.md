# Multi-Score Evaluation Script

## Dependencies

Check the `requirements.txt` for dependencies needed.

## Usage

To compute the Multi-Score of your test cases, run the `Multi-Score.py` script which receives as input the following parameters:

```
usage: Multi-Score.py [-h] -R REFERENCE -H1 HYPOTHESIS1 -H2 HYPOTHESIS2 -H3 HYPOTHESIS3 [-nr NUM_REFS] [-nc NCORDER] [-nw NWORDER] [-b BETA]

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

An example on how to run to the evaluation script is available in `example.sh`. For instance, to obtain the Multi-Score, BLEU score, ChrF++ score, Self-BLEU score, simply run the following command:

```
python3 Multi-Score.py -R data/references/test_reference -H1 data/predictions/prediction_01.txt -H2 data/predictions/prediction_02.txt -H3 data/predictions/prediction_03.txt
```

### Multiple References

In case of multiple references, they have to be stored in separated files and named reference0, reference1, reference2, etc.# Multi-Score
