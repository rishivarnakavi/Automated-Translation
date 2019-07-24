import math
from translation import bing
from nltk.compat import Counter
from nltk.util import ngrams

count=0
class BLEU(object):
    @staticmethod
    def compute(candidate, references, weights):
        candidate = [c.lower() for c in candidate]
        references = [[r.lower() for r in reference]]
        
        p_ns = (BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)

        bp = BLEU.penalty(candidate, references)
        return bp * math.exp(s)

    @staticmethod
    def modified_precision(candidate, references, n):
        counts = Counter(ngrams(candidate, n))

        if not counts:
            return 0

        max_counts = {}
        for reference in references:
            reference_counts = Counter(ngrams(reference, n))
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

        clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

        return sum(clipped_counts.values()) / sum(counts.values())

    @staticmethod
    def penalty(candidate, references):
        c = len(candidate)
        r = min(abs(len(r) - c) for r in references)

        if c > r:
            return 1
        else:
            return math.exp(1 - r / c)


if __name__ == "__main__":

    #Reading the English file
    with open("5_en.txt", "r", encoding="utf8") as file:
        for line in file:
            if line:
                line = line.rstrip()
                line = line.lower()
                candidate=[]
                word = line.split()
                candidate.append(word)
                cand_length=len(candidate)

    # Reading the Spanish file
    with open("5_es.txt", "r", encoding="utf8") as file1:
        for line1 in file1:
            if line1:
                count=count+1
                print("Line number = "+str(count))
                line1 = line1.rstrip()
                line1 = line1.lower()
                reference=[]
                target = 'en'
                line1 = line1.split()
                print(line1)
                for sub1 in line1:
                    # word to word translation from spanish to english
                    try:
                        english = (bing(sub1, dst='en'))
                    except:
                        pass
                    reference.append(english)
                print(reference)
    score=[]
    for (cand,ref) in zip(candidate,reference):
        score.append(BLEU.compute(cand,ref,[0.25,0.25,0.25,0.25]))
    bleu=sum(score) / float(len(score))
    print(bleu)