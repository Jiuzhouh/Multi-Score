from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from scipy.optimize import linear_sum_assignment
import numpy as np
import string
import codecs
import copy
import nltk
import argparse

def separate_characters(line):
    return list(line.strip().replace(" ", ""))

def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1]
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)

    return tokenized

def ngram_counts(wordList, order):
    counts = defaultdict(lambda: defaultdict(float))
    nWords = len(wordList)
    for i in range(nWords):
        for j in range(1, order+1):
            if i+j <= nWords:
                ngram = tuple(wordList[i:i+j])
                counts[j-1][ngram]+=1

    return counts

def ngram_matches(ref_ngrams, hyp_ngrams):
    matchingNgramCount = defaultdict(float)
    totalRefNgramCount = defaultdict(float)
    totalHypNgramCount = defaultdict(float)

    for order in ref_ngrams:
        for ngram in hyp_ngrams[order]:
            totalHypNgramCount[order] += hyp_ngrams[order][ngram]
        for ngram in ref_ngrams[order]:
            totalRefNgramCount[order] += ref_ngrams[order][ngram]
            if ngram in hyp_ngrams[order]:
                matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])

    return matchingNgramCount, totalRefNgramCount, totalHypNgramCount

def ngram_precrecf(matching, reflen, hyplen, beta):
    ngramPrec = defaultdict(float)
    ngramRec = defaultdict(float)
    ngramF = defaultdict(float)

    factor = beta**2

    for order in matching:
        if hyplen[order] > 0:
            ngramPrec[order] = matching[order]/hyplen[order]
        else:
            ngramPrec[order] = 1e-16
        if reflen[order] > 0:
            ngramRec[order] = matching[order]/reflen[order]
        else:
            ngramRec[order] = 1e-16
        denom = factor*ngramPrec[order] + ngramRec[order]
        if denom > 0:
            ngramF[order] = (1+factor)*ngramPrec[order]*ngramRec[order] / denom
        else:
            ngramF[order] = 1e-16

    return ngramF, ngramRec, ngramPrec

def parse(refs_path, hyps_path, num_refs, lng='en'):
    # print('STARTING TO PARSE INPUTS...')
    # references
    references = []
    for i in range(num_refs):
        fname = refs_path + str(i) if num_refs > 1 else refs_path
        with codecs.open(fname, 'r', 'utf-8') as f:
            texts = f.read().split('\n')
            for j, text in enumerate(texts):
                if len(references) <= j:
                    references.append([text])
                else:
                    references[j].append(text)

    # references tokenized
    references_tok = copy.copy(references)
    for i, refs in enumerate(references_tok):
        if lng == 'ru':
            references_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
        else:
            references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

    # hypothesis
    with codecs.open(hyps_path, 'r', 'utf-8') as f:
        hypothesis = f.read().split('\n')

    # hypothesis tokenized
    hypothesis_tok = copy.copy(hypothesis)
    if lng == 'ru':
        hypothesis_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in hypothesis_tok]
    else:
        hypothesis_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in hypothesis_tok]

    # print('FINISHING TO PARSE INPUTS...')
    return references, references_tok, hypothesis, hypothesis_tok

def computeChrF(fpRef, fpHyp, nworder, ncorder, beta):
    norder = float(nworder + ncorder)

    # initialisation of document level scores
    totalMatchingCount = defaultdict(float)
    totalRefCount = defaultdict(float)
    totalHypCount = defaultdict(float)
    totalChrMatchingCount = defaultdict(float)
    totalChrRefCount = defaultdict(float)
    totalChrHypCount = defaultdict(float)
    averageTotalF = 0.0

    nsent = 0
    totalSentF = []
    for hline, rline in zip(fpHyp, fpRef):
        nsent += 1

        # preparation for multiple references
        maxF = -1.0

        hypNgramCounts = ngram_counts(separate_punctuation(hline), nworder)
        hypChrNgramCounts = ngram_counts(separate_characters(hline), ncorder)

        # going through multiple references

        refs = rline.split("*#")

        sentF_list = []
        for ref in refs:
            refNgramCounts = ngram_counts(separate_punctuation(ref), nworder)
            refChrNgramCounts = ngram_counts(separate_characters(ref), ncorder)

            # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
            matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = ngram_matches(refNgramCounts, hypNgramCounts)
            matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = ngram_matches(refChrNgramCounts, hypChrNgramCounts)

            # n-gram f-scores, recalls and precisions
            ngramF, ngramRec, ngramPrec = ngram_precrecf(matchingNgramCounts, totalRefNgramCount, totalHypNgramCount, beta)
            chrNgramF, chrNgramRec, chrNgramPrec = ngram_precrecf(matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount, beta)

            sentRec  = (sum(chrNgramRec.values())  + sum(ngramRec.values()))  / norder
            sentPrec = (sum(chrNgramPrec.values()) + sum(ngramPrec.values())) / norder
            sentF    = (sum(chrNgramF.values())    + sum(ngramF.values()))    / norder
            sentF_list.append(sentF)

            if sentF > maxF:
                maxF = sentF
                bestMatchingCount = matchingNgramCounts
                bestRefCount = totalRefNgramCount
                bestHypCount = totalHypNgramCount
                bestChrMatchingCount = matchingChrNgramCounts
                bestChrRefCount = totalChrRefNgramCount
                bestChrHypCount = totalChrHypNgramCount

        totalSentF.append(sentF_list)

        # collect document level ngram counts
        for order in range(nworder):
            totalMatchingCount[order] += bestMatchingCount[order]
            totalRefCount[order] += bestRefCount[order]
            totalHypCount[order] += bestHypCount[order]
        for order in range(ncorder):
            totalChrMatchingCount[order] += bestChrMatchingCount[order]
            totalChrRefCount[order] += bestChrRefCount[order]
            totalChrHypCount[order] += bestChrHypCount[order]

        averageTotalF += maxF

    # total precision, recall and F (aritmetic mean of all ngrams)
    totalNgramF, totalNgramRec, totalNgramPrec = ngram_precrecf(totalMatchingCount, totalRefCount, totalHypCount, beta)
    totalChrNgramF, totalChrNgramRec, totalChrNgramPrec = ngram_precrecf(totalChrMatchingCount, totalChrRefCount, totalChrHypCount, beta)
    totalF = (sum(totalChrNgramF.values()) + sum(totalNgramF.values())) / norder

    return totalSentF, totalF

def ChrF_Score(references, hypotheses, num_refs, nworder, ncorder, beta):
    hyps_tmp, refs_tmp = 'hypothesis_chrF', 'reference_chrF'

    references_, hypothesis_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append(refs_)
            hypothesis_.append(hypothesis[i])

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis_))

    linear_references = []
    for refs in references_:
        linear_references.append('*#'.join(refs[:num_refs]))

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    rtxt = codecs.open(refs_tmp, 'r', 'utf-8')
    htxt = codecs.open(hyps_tmp, 'r', 'utf-8')

    try:
        totalSentF, totalF = computeChrF(rtxt, htxt, nworder, ncorder, beta)
    except:
        print('ERROR ON COMPUTING ChrF.')
        totalSentF = []
        totalF = -1
    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass

    return totalSentF, totalF

def MultiScore(totalSentScore_1, totalSentScore_2, totalSentScore_3):
    senTotalScore= []
    for i in range(len(totalSentScore_1)):
        score1 = totalSentScore_1[i]
        score2 = totalSentScore_2[i]
        score3 = totalSentScore_3[i]
        minus_score1 = []
        minus_score2 = []
        minus_score3 = []
        for j in range(len(score1)):
            minus_score1.append(-score1[j])
            minus_score2.append(-score2[j])
            minus_score3.append(-score3[j])
        cost = np.array([minus_score1, minus_score2, minus_score3])
        row_ind, col_ind = linear_sum_assignment(cost)
        minus_BLEU = cost[row_ind, col_ind].sum()/len(col_ind)
        senTotalScore.append(-minus_BLEU)
    averageSentBLEU = sum(senTotalScore)/len(senTotalScore)

    return averageSentBLEU

def BLEU_Score(references, hypothesis):
    # check for empty lists
    references_, hypothesis_ = [], []
    totalScores = []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append([ref.split() for ref in refs_])
            hypothesis_.append(hypothesis[i].split())

    chencherry = SmoothingFunction()
    corpus_bleu_score = corpus_bleu(references_, hypothesis_, smoothing_function=chencherry.method3)

    for i in range(len(hypothesis_)):
        scores = []
        for j in range(len(references_[i])):
            scores.append(sentence_bleu([references_[i][j]], hypothesis_[i], smoothing_function=chencherry.method3))
        totalScores.append(scores)

    return totalScores, corpus_bleu_score

def Self_BLEU_Score(references, hypothesis):
    # check for empty lists
    references_, hypothesis_ = [], []
    totalScores = []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append([ref.split() for ref in refs_])
            hypothesis_.append(hypothesis[i].split())

    chencherry = SmoothingFunction()

    for i in range(len(hypothesis_)):
        scores = []
        for j in range(len(references_[i])):
            scores.append(sentence_bleu([references_[i][j]], hypothesis_[i], smoothing_function=chencherry.method3))
        totalScores.append(sum(scores)/len(scores))
    return totalScores

def get_self_reference(hypothesis_tok1, hypothesis_tok2, hypothesis_tok3):
    self_reference = []
    for i in range(len(hypothesis_tok1)):
        ref = []
        ref.append(hypothesis_tok1[i])
        ref.append(hypothesis_tok2[i])
        ref.append(hypothesis_tok3[i])
        self_reference.append(ref)

    return self_reference


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-R", "--reference", help="reference translation", required=True)
    argParser.add_argument("-H1", "--hypothesis1", help="hypothesis translation", required=True)
    argParser.add_argument("-H2", "--hypothesis2", help="hypothesis translation", required=True)
    argParser.add_argument("-H3", "--hypothesis3", help="hypothesis translation", required=True)
    argParser.add_argument("-nr", "--num_refs", help="number of references", type=int, default=3)
    argParser.add_argument("-nc", "--ncorder", help="chrF metric: character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nw", "--nworder", help="chrF metric: word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-b", "--beta", help="chrF metric: beta parameter (default=2)", type=float, default=2.0)

    args = argParser.parse_args()

    # print('READING INPUTS...')
    refs_path = args.reference
    hyps_path1 = args.hypothesis1
    hyps_path2 = args.hypothesis2
    hyps_path3 = args.hypothesis3
    num_refs = args.num_refs
    nworder = args.nworder
    ncorder = args.ncorder
    beta = args.beta
    # print('FINISHING TO READ INPUTS...')

    #refs_path='data/references/reference'
    #hyps_path1='data/predictions/prediction_01.txt'
    #hyps_path2='data/predictions/prediction_02.txt'
    #hyps_path3='data/predictions/prediction_03.txt'

    references, references_tok, hypothesis, hypothesis_tok1 = parse(refs_path, hyps_path1, num_refs)
    totalSentBLEU_1, totalBLEU_1 = BLEU_Score(references_tok, hypothesis_tok1)
    totalSentChrF_1, totalChrF_1 = ChrF_Score(references, hypothesis, num_refs, nworder, ncorder, beta)

    references, references_tok, hypothesis, hypothesis_tok2 = parse(refs_path, hyps_path2, num_refs)
    totalSentBLEU_2, totalBLEU_2 = BLEU_Score(references_tok, hypothesis_tok2)
    totalSentChrF_2, totalChrF_2 = ChrF_Score(references, hypothesis, num_refs, nworder, ncorder, beta)

    references, references_tok, hypothesis, hypothesis_tok3 = parse(refs_path, hyps_path3, num_refs)
    totalSentBLEU_3, totalBLEU_3 = BLEU_Score(references_tok, hypothesis_tok3)
    totalSentChrF_3, totalChrF_3 = ChrF_Score(references, hypothesis, num_refs, nworder, ncorder, beta)

    #compute MultiScore_ChrF and ChrF
    MultiScore_ChrF = MultiScore(totalSentChrF_1, totalSentChrF_2, totalSentChrF_3)
    ChrF = (totalChrF_1+totalChrF_2+totalChrF_3)/3

    #compute MultiScore_BLEU and BLEU
    MultiScore_BLEU = MultiScore(totalSentBLEU_1, totalSentBLEU_2, totalSentBLEU_3)
    BLEU = (totalBLEU_1+totalBLEU_2+totalBLEU_3)/3

    self_reference = get_self_reference(hypothesis_tok1, hypothesis_tok2, hypothesis_tok3)
    total_selfBLEU_1 = Self_BLEU_Score(self_reference, hypothesis_tok1)
    total_selfBLEU_2 = Self_BLEU_Score(self_reference, hypothesis_tok2)
    total_selfBLEU_3 = Self_BLEU_Score(self_reference, hypothesis_tok3)
    Self_BLEU = (sum(total_selfBLEU_1)/len(total_selfBLEU_1)+sum(total_selfBLEU_2)/len(total_selfBLEU_2)+sum(total_selfBLEU_3)/len(total_selfBLEU_3))/3

    print('BLEU:', round(BLEU*100, 2))
    print('ChrF++:', round(ChrF*100, 2))
    print('MultiScore-BLEU:', round(MultiScore_BLEU*100, 2))
    print('MultiScore-ChrF++:', round(MultiScore_ChrF*100, 2))
    print('Self-BLEU:', round(Self_BLEU*100, 2))
