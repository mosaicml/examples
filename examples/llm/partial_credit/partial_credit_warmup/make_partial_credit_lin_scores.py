
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from llm.tokenizer import TOKENIZER_REGISTRY
import string
import re
from tqdm import tqdm
import pickle
import time
import os
from nltk.corpus import wordnet_ic
import nltk

IC = nltk.corpus.wordnet_ic.ic('ic-semcor.dat')

MATCH_SCORE_CUTOFF = 0.5
FUZZY_PENALTY = 0.5
WORDNET_WORDS = [x for x in wn.all_lemma_names()]
MAX_PARTIAL_CREDIT = 100

PARTIAL_CREDIT_MIN_SCORE = 0.5


def fuzzy_match_gpt_token_to_wordnet_tokens(token):
    filtered_wn_words = {x: 1-len(re.sub(token, '', x))/len(x) for x in WORDNET_WORDS if token in x}
    if len(filtered_wn_words) == 0:
        return None
    best_match = sorted(filtered_wn_words, key=filtered_wn_words.get, reverse=True)[0]
    best_match_score = filtered_wn_words[best_match]
    if best_match_score > MATCH_SCORE_CUTOFF:
        return {"token": token, "synsets": retrieve_wordnet_synsets(best_match), "score": FUZZY_PENALTY * best_match_score}
    else:
        return None

def retrieve_wordnet_synsets(token):
    return wn.synsets(token)


def mean(x):
    return sum(x) / len(x)



WUP_SIM_CACHE = {}



def calculate_token_pair_lin_score(token_id1, token_id2, tokens_to_synsets, synset_score_aggregation):
    global WUP_SIM_CACHE

    sns1, score1 = tokens_to_synsets[token_id1]['synsets'], tokens_to_synsets[token_id1]['score']
    sns2, score2 = tokens_to_synsets[token_id2]['synsets'], tokens_to_synsets[token_id2]['score']

    sims = []
    for sn1 in sns1:
        for sn2 in sns2:
            if (sn1.name(), sn2.name()) in WUP_SIM_CACHE:
                sims.append(WUP_SIM_CACHE[(sn1.name(), sn2.name())])
               
            elif (sn2.name(), sn1.name()) in WUP_SIM_CACHE:
                sims.append(WUP_SIM_CACHE[(sn2.name(), sn1.name())])
            else:
                try:
                    score = sn1.lin_similarity(sn2, IC)
                except:
                    score = 0
                sims.append(score)
              
                if len(WUP_SIM_CACHE) > 10_000_000:
                    WUP_SIM_CACHE = {}
                
                WUP_SIM_CACHE[(sn1.name(), sn2.name())] = score
               


    if len(sims) == 0:
        return 0

    reduced_score = synset_score_aggregation(sims)
    return reduced_score * score1 * score2

def construct_partial_credit_map(tokenizer, tokens_to_synsets, output_file, synset_score_aggregation=max):
    global WUP_SIM_CACHE

    if os.path.exists(output_file):
        with open(output_file, 'rb') as handle:
            final_result = pickle.load(handle)
    else:
        final_result = {}

    if os.path.exists("llm/speedups/artifacts/full_lin_map.pkl"):
        with open("llm/speedups/artifacts/full_lin_map.pkl", 'rb') as handle:
            partial_credit_map = pickle.load(handle)
    else:
        partial_credit_map = {}

   
    WUP_SIM_CACHE = {}

    print(f"Reloaded {len(final_result)}")
    print(f"Reloaded {len(partial_credit_map)}")

    got_to_skip = 0
    comparison_count = 0
    counter = 0
    for token1 in tqdm(range(len(tokenizer.tokenizer.vocab.values()))):
        if token1 in final_result:
            print(f"{token1} already computed")
            continue
        partial_credit_map[token1] = {}

        for token2 in range(len(tokenizer.tokenizer.vocab.values())):
            comparison_count += 1
            if token1 == token2:
                partial_credit_map[token1][token2] = 1.0
            elif token2 in partial_credit_map:
                if token1 in partial_credit_map[token2]:
                    partial_credit_map[token1][token2] = partial_credit_map[token2][token1]
            else:
                got_to_skip += 1
                score = calculate_token_pair_lin_score(
                    token1, token2, tokens_to_synsets, synset_score_aggregation
                )
                if score > PARTIAL_CREDIT_MIN_SCORE:
                    partial_credit_map[token1][token2] = score
     

        subset = {k: v for k,v in partial_credit_map[token1].items() if v > PARTIAL_CREDIT_MIN_SCORE}
        final_result[token1] = [(tok, subset[tok])
            for tok in sorted(subset, key=subset.get, reverse=True)[0:MAX_PARTIAL_CREDIT]
        ]
     
        if counter % 50 == 0 and counter > 0:
            print(f"Got to skip {(comparison_count - got_to_skip) / comparison_count}")
            
            with open(output_file, 'wb') as handle:
                pickle.dump(final_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if counter % 100 == 0 and counter >= 500:
                with open("llm/speedups/artifacts/full_lin_map.pkl", "wb") as handle:
                    pickle.dump(partial_credit_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
        counter += 1
    
    

        

        

def make_partial_credit_scores_from_wordnet(tokenizer, output_file, whole_words_only):
    counter = 0
    total = 0
    tokens_to_synsets = {}
    for token_id in tqdm(tokenizer.tokenizer.vocab.values()):
        token = tokenizer.tokenizer.decode(token_id).strip().lower().translate(str.maketrans('', '', string.punctuation))
        synsets = retrieve_wordnet_synsets(token)
        tokens_to_synsets[token_id] = {"token": token, "synsets": [], "score": 0}
        addl_predicate = True if not whole_words_only else tokenizer.tokenizer.decode(token_id).startswith(" ") and len(token) > 2
        if len(synsets) > 0 and addl_predicate:
            tokens_to_synsets[token_id] = {"token": token, "synsets": synsets, "score": 1}
            counter +=1
        else:
            pass
    

        total += 1
    print(f"{counter/total} percent of tokens were in wordnet")
    
    construct_partial_credit_map(tokenizer, tokens_to_synsets, output_file=output_file, synset_score_aggregation=max)

    



if __name__=="__main__":
    tokenizer = TOKENIZER_REGISTRY["hftokenizer"]("bert-base-uncased", 2048)
    fn = f"llm/speedups/artifacts/wordnet_lin_top_{MAX_PARTIAL_CREDIT}_whole_word_only_bert_tokenizer.pkl"
    make_partial_credit_scores_from_wordnet(tokenizer, output_file=fn, whole_words_only=False)
