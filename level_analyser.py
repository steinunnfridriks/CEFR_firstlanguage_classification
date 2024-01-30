"""
This program analyses the features of each CEFR level based on the small, accompanying dataset. 
This analysis is independent of any specific classification model but can be used as a basis
for feature engineering in said models.
"""

import pandas as pd
import numpy as np
import torch
import pos
from nltk import word_tokenize
from reynir import Greynir
from tokenizer import split_into_sentences

df = pd.read_csv('cefrdata.csv')

texts = df.query("cefrlevel=='C2'")["text"] # works with one level at a time (could be changed to work with all at once)

device = torch.device("cuda")
tagger = pos.Tagger = torch.hub.load( # The POS package is based on ABLTagger (see https://github.com/cadia-lvl/POS)
    repo_or_dir="cadia-lvl/POS",
    model="tag",
    device=device,
    force_reload=False,
    force_download=False,
)

average_subordinate = 0 # Average porportion of sentences that contain subordinate clauses (based on total no of sentences) per text per level 
average_comp_sub = 0 # Average porportion of sentences that contain compliment clauses (based on total no of sentences) per text per level 
average_que_sub = 0 # Average porportion of sentences that contain question clauses (based on total no of sentences) per text per level 
average_rel_sub = 0 # Average porportion of sentences that contain relative clauses (based on total no of sentences) per text per level 
average_temp_sub = 0 # Average porportion of sentences that contain adverbial temporal phrases (based on total no of sentences) per text per level 
average_purp_sub = 0 # Average porportion of sentences that contain adverbial purpose phrases (based on total no of sentences) per text per level
average_aknow_sub = 0 # Average porportion of sentences that contain adverbial aknowledgement phrases (based on total no of sentences) per text per level
average_cons_sub = 0 # Average porportion of sentences that contain adverbial consequence phrases (based on total no of sentences) per text per level
average_caus_sub = 0 # Average porportion of sentences that contain adverbial causal phrases (based on total no of sentences) per text per level
average_cond_sub = 0 # Average porportion of sentences that contain adverbial conditional phrases (based on total no of sentences) per text per level
average_compar_sub = 0 # Average porportion of sentences that contain adverbial comparative phrases (based on total no of sentences) per text per level

average_total_adjectives = 0 # Average porportion of adjectives (based on total no of words) per text per level
average_total_nouns = 0 # Average porportion of nouns (based on total no of words) per text per level
average_total_verbs = 0 # Average porportion of verbs (based on total no of words) per text per level
average_total_pronouns = 0 # Average porportion of pronouns (based on total no of words) per text per level
average_total_particles = 0 # Average porportion of particles [and adverbs, POS c and a in ABLTagger] (based on total no of words) per text per level
average_subjunctive = 0 # Average porportion of verbs in the subjunctive (based on total no of verbs) per text per level
average_middle_voice = 0 # Average porportion of verbs in the middle voice (based on total no of verbs) per text per level
average_passive_voice = 0 # Average porportion of verbs in the past participle [usually passive voice] (based on total no of verbs) per text per level
average_superlative_oblique = 0 # Average porportion of adjectives in the superlative that have an oblique case (based on total no of adjectives) per text per level
average_definite_oblique = 0 # Average porportion of definite nouns that have an oblique case (based on total no of nouns) per text per level
average_interrogatives_oblique = 0 # Average porportion of interrogatives that have an oblique case (based on total no of interrogatives) per text per level
average_superlative = 0 # Average porportion of adjectives in the superlative (based on total no of adjectives) per text per level
average_1p_pronouns = 0 # Average porportion of pronouns in the first person (based on total no of pronouns) per text per level
average_past = 0 # Average porportion of verbs in the past tense (based on total no of verbs) per text per level 
averagelongwordscore = 0 # Average porportion of words with more than 6 letters (based on total no of words) per text per level

average_total_unique_words = 0 # Average porportion of unique word forms per text per level
average_total_unique_verbforms = 0 # Average porportion of unique verb conjugations per text per level
average_total_unique_nounforms = 0 # Average porportion of unique noun conjugations per text per level
average_total_unique_adjectiveforms = 0 # Average porportion of unique adjective conjugations per text per level

count = 1
for i in texts:
    print("Working on text", count, "out of", len(texts))
    subordinates = 0 # Counter for the presence of subordinate clauses in sentences in each text
    subjunctive = 0 # Counter for verbs in the subjunctive in each text
    middle = 0 # Counter for verbs in the middle voice in each text
    passive = 0 # Counter for verbs in the past participle (usually passive voice) in each text
    adj_superlative_oblique = 0 # Counter for adjectives in the superlative in an oblique case in each text
    noun_definite_oblique = 0 # Counter for definite nouns in an oblique case in each text
    interrogatives_oblique = 0 # Counter for interrogatives in an oblique case in each text
    superlative = 0 # Counter for adjectives in the superlative in each text
    pronoun_1p = 0 # Counter for pronouns in the first person in each text
    past_verb = 0 # Counter for verbs in the past tense (not past participle) in each text
    total_adjectives = 0 # Counter for all adjectives in each text 
    total_verbs = 0 # Counter for all verbs in each text
    total_nouns = 0 # Counter for all nouns in each text
    total_particles = 0 # Counter for all particles [and adverbs] in each text
    total_interrogatives = 0 # Counter for all interrogatives in each text
    total_pronouns = 0 # Counter for all pronouns in each text
    unique_verbforms = [] # List keeping track of all unique POS tags for verbs in each text
    unique_nounforms = [] # List keeping track of all unique POS tags for nouns in each text
    unique_adjectiveforms = [] # List keeping track of all unique POS tags for adjectives in each text
    average_unique_words = 0 # Counter for unique word forms in each text
    average_unique_verbforms = 0 # Counter for unique unique verb conjugations in each text 
    average_unique_nounforms = 0 # Counter for unique unique noun conjugations in each text 
    average_unique_adjectiveforms = 0 # Counter for unique unique adjective conjugations in each text

    comp_subordinates = 0 # Counter for the presence of compliment clauses in sentences in each text
    que_subordinates = 0 # Counter for the presence of question clauses in sentences in each text
    rel_subordinates = 0 # Counter for the presence of relative clauses in sentences in each text
    temp_subordinates = 0 # Counter for the presence of adverbial temporal phrases in sentences in each text
    purp_subordinates = 0 # Counter for the presence of adverbial purpose phrases in sentences in each text
    aknow_subordinates = 0 # Counter for the presence of adverbial aknowledgement phrases in sentences in each text
    cons_subordinates = 0 # Counter for the presence of adverbial consequence phrases in sentences in each text
    caus_subordinates = 0 # Counter for the presence of adverbial causal phrases in sentences in each text
    cond_subordinates = 0 # Counter for the presence of adverbial conditional phrases in sentences in each text
    compar_subordinates = 0 # Counter for the presence of adverbial comparative phrases in sentences in each text

    g = Greynir() # Using Greynir to parse the text and analyse the presence of subordinate clauses
    sents = split_into_sentences(i) 
    anysent = 0 # Counter to keep track of any sentence that gets parsed properly by Greynir in each text

    for sent in sents:
        parsed = g.parse_single(sent)
        if parsed and parsed.tree: # If Greynir parses the sentence properly
            anysent += 1
            try:
                if "CP-QUE" in parsed.tree.flat or "CP-THT" in parsed.tree.flat or "CP-REL" in parsed.tree.flat or "CP-ADV-TEMP" in parsed.tree.flat or "CP-ADV-PURP" in parsed.tree.flat or "CP-ADV-ACK" in parsed.tree.flat or "CP-ADV-CONS" in parsed.tree.flat or "CP-ADV-CAUSE" in parsed.tree.flat or "CP-ADV-COND" in parsed.tree.flat or "CP-ADV-CMP" in parsed.tree.flat: # any type subordinate clause is present in the sentence
                    subordinates += 1

                if "CP-THT" in parsed.tree.flat: # Compliment clause ("Páll veit -að kötturinn kemur heim-")
                    comp_subordinates += 1
                if "CP-QUE" in parsed.tree.flat: # Question subclause (Páll spurði -hvaða stjaka hún vildi-)
                    que_subordinates += 1
                if "CP-REL" in parsed.tree.flat: # Relative clause (Páll, -sem kom inn-, klappaði kettinum)
                    rel_subordinates += 1
                if "CP-ADV-TEMP" in parsed.tree.flat: # Adverbial temporal phrase (Páll fór út -á meðan kötturinn mjálmaði-)
                    temp_subordinates += 1
                if "CP-ADV-PURP" in parsed.tree.flat: # Adverbial purpose phrase (Fuglinn flaug -til þess að ná sér í mat-)
                    purp_subordinates += 1
                if "CP-ADV-ACK" in parsed.tree.flat: # Adverbial aknowledgement phrase (Páll fór út -þó að hann væri þreyttur-)
                    aknow_subordinates += 1
                if "CP-ADV-CONS" in parsed.tree.flat: # Adverbial consequence phrase (Páll fór út -þannig að hann er þreyttur-)
                    cons_subordinates += 1
                if "CP-ADV-CAUSE" in parsed.tree.flat: # Adverbial causal phrase (Páll fór út -þar sem hann er þreyttur-)
                    caus_subordinates += 1
                if "CP-ADV-COND" in parsed.tree.flat: # Adverbial conditional phrase (Páll færi út -ef hann gæti-)
                    cond_subordinates += 1
                if "CP-ADV-CMP" in parsed.tree.flat: # Adverbial comparative phrase (Páll hleypur hraðar -en kötturinn-)
                    compar_subordinates += 1
            except AttributeError:
                continue

    text = word_tokenize(i) # Tokenizing the text to keep track of total number of words per text
    average_unique_words += len(set(text)) # Counting the unique word forms of each text, averaged below by total number of texts
    try:
        tags = tagger.tag_sent(text) # Getting the POS tags for each text
    except AssertionError: # Avoiding an error from POS saying that max_length is wrong
        continue
    tagged = list(zip(text, tags)) # Matching each word with the corresponding tag
    longwordscore = 0 # Keeping track of how many words exceed 6 letters
    
    for word, tag in tagged:
        
        if tag.startswith("n"): # Nouns
            total_nouns += 1
            if tag not in unique_nounforms:
                unique_nounforms.append(tag)
        elif tag.startswith("l"): # Adjectives
            total_adjectives += 1
            if tag not in unique_adjectiveforms:
                unique_adjectiveforms.append(tag)
        elif tag.startswith("s"): # Verbs
            total_verbs += 1
            if tag not in unique_verbforms:
                unique_verbforms.append(tag)
        elif tag.startswith("fs"): # Interrogatives
            total_interrogatives += 1
        elif tag.startswith("fp"): # Pronouns
            total_pronouns += 1
        elif tag.startswith("c") or tag.startswith("a"): # Conjunctions and adverbs/prepositions/interjections
            total_particles += 1

        if len(word) > 6:
            longwordscore += 1

        if tag.startswith("sv"): # Verbs in the subjunctive
            subjunctive += 1
        elif tag.startswith("s") and "m" in tag: # Verbs in the middle voice
            middle += 1
        elif tag.startswith("sþ"): # Verbs in the past participle (usually passive voice)
            passive += 1
        elif tag.startswith("l") and "e" in tag and not "n" in tag: # Adjectives in the superlative with an oblique case
            adj_superlative_oblique+= 1
        elif tag.startswith("n") and "g" in tag and tag[3] != "n": # Definite nouns with an oblique case
            noun_definite_oblique += 1
        elif tag.startswith("fs") and not "n" in tag: # Interrogatives with an oblique case
            interrogatives_oblique += 1
        elif tag.startswith("l") and tag[5] == "e": # Adjectives in the superlative
            superlative += 1
        elif tag.startswith("fp1"): # Pronouns in the first person
            pronoun_1p += 1           
        elif tag.startswith("s") and tag[-1] == "þ": # Verbs in the past tense (not past participle)
            past_verb += 1

    if anysent != 0: # Avoiding division by zero in case no sentence was parsed
        average_subordinate += (subordinates / anysent)
        average_comp_sub += (comp_subordinates / anysent)
        average_que_sub += (que_subordinates / anysent)
        average_rel_sub += (rel_subordinates / anysent)
        average_temp_sub += (temp_subordinates / anysent)
        average_purp_sub += (purp_subordinates / anysent)
        average_aknow_sub += (aknow_subordinates / anysent)
        average_cons_sub += (cons_subordinates / anysent)
        average_caus_sub += (caus_subordinates / anysent)
        average_cond_sub += (cond_subordinates / anysent)
        average_compar_sub += (compar_subordinates / anysent)
    else:
        average_subordinate += 0
        average_comp_sub += 0
        average_que_sub += 0
        average_rel_sub += 0
        average_temp_sub += 0
        average_purp_sub += 0
        average_aknow_sub += 0
        average_cons_sub += 0
        average_caus_sub += 0
        average_cond_sub += 0
        average_compar_sub += 0

    if total_verbs != 0: # In all cases, avoiding division by zero in case no words of the corresponding POS was counted
        average_subjunctive += (subjunctive / total_verbs)
        average_middle_voice += (middle / total_verbs)
        average_passive_voice += (passive / total_verbs)
        average_past += (past_verb / total_verbs)
        average_unique_verbforms += len(unique_verbforms)
        average_total_unique_verbforms += (average_unique_verbforms / total_verbs)
    else:
        average_subjunctive += 0
        average_middle_voice += 0
        average_passive_voice += 0
        average_past += 0
        average_unique_verbforms += 0
        average_total_unique_verbforms += 0

    if total_adjectives != 0:
        average_superlative_oblique += (adj_superlative_oblique/ total_adjectives)
        average_superlative += (superlative / total_adjectives)
        average_unique_adjectiveforms += len(unique_adjectiveforms)
        average_total_unique_adjectiveforms += (average_unique_adjectiveforms / total_adjectives)
    else:
        average_superlative_oblique += 0
        average_unique_adjectiveforms += 0
        average_total_unique_adjectiveforms += 0

    if total_nouns != 0:
        average_definite_oblique += (noun_definite_oblique / total_nouns)
        average_unique_nounforms += len(unique_nounforms)
        average_total_unique_nounforms += (average_unique_nounforms / total_nouns)
    else:
        average_definite_oblique += 0
        average_unique_nounforms += 0
        average_total_unique_nounforms += 0

    if total_interrogatives != 0:
        average_interrogatives_oblique += (interrogatives_oblique / total_interrogatives)
    else:
        average_interrogatives_oblique += 0

    if total_pronouns != 0:
        average_1p_pronouns += (pronoun_1p / total_pronouns)
    else:
        average_1p_pronouns += 0
    
    averagelongwordscore += (longwordscore / len(text))
    average_total_adjectives += (total_adjectives / len(text))
    average_total_nouns += (total_nouns / len(text))
    average_total_verbs += (total_verbs / len(text))
    average_total_pronouns += (total_pronouns / len(text))
    average_total_particles += (total_particles / len(text))
    average_total_unique_words += (average_unique_words / len(text))
    count += 1          

print("__________")
print()
print("Average proportion of sentences with subordinate clauses per text in the level:", average_subordinate / len(texts))
print("Average proportion of sentences with compliment clauses per text in the level:", average_comp_sub / len(texts))
print("Average proportion of sentences with question clauses per text in the level:", average_que_sub / len(texts))
print("Average proportion of sentences with relative clauses per text in the level:", average_rel_sub / len(texts))
print("Average proportion of sentences with adverbial temporal phrases per text in the level:", average_temp_sub / len(texts))
print("Average proportion of sentences with adverbial purpose phrases per text in the level:", average_purp_sub / len(texts))
print("Average proportion of sentences with adverbial aknowledgement phrases per text in the level:", average_aknow_sub / len(texts))
print("Average proportion of sentences with adverbial consequence phrases per text in the level:", average_cons_sub / len(texts))
print("Average proportion of sentences with adverbial causal phrases per text in the level:", average_caus_sub / len(texts))
print("Average proportion of sentences with adverbial conditional phrases per text in the level:", average_cond_sub / len(texts))
print("Average proportion of sentences with adverbial comparative phrases per text in the level:", average_compar_sub / len(texts))

print("Average proportion of adjectives per text in the level:",average_total_adjectives / len(texts))
print("Average proportion of nouns per text in the level:", average_total_nouns / len(texts))
print("Average proportion of verbs per text in the level:", average_total_verbs / len(texts))
print("Average proportion of pronouns per text in the level:", average_total_pronouns / len(texts))
print("Average proportion of particles (and adverbs) per text in the level:", average_total_particles / len(texts))

print("Average proportion of verbs in the subjunctive per text in the level:", average_subjunctive / len(texts))
print("Average proportion of verbs in the middle voice per text in the level:", average_middle_voice / len(texts))
print("Average proportion of verbs in the past participle (usually passive voice) per text in the level:", average_passive_voice / len(texts))
print("Average proportion of adjectives in the superlative with an oblique case per text in the level:", average_superlative_oblique / len(texts))
print("Average proportion of definite nouns with an oblique case per text in the level:", average_definite_oblique / len(texts))
print("Average proportion of interrogatives with an oblique case per text in the level:", average_interrogatives_oblique / len(texts))
print("Average proportion of adjectives in the superlative per text in the level:", average_superlative / len(texts))
print("Average proportion of pronouns in the first person per text in the level:", average_1p_pronouns / len(texts))
print("Average proportion of verbs in the past tense (not past participle) per text in the level:", average_past / len(texts))

print("Average proportion of words with more than 6 letters per text in the level:", averagelongwordscore / len(texts))

print("Average proportion of unique word forms per text per level:", average_total_unique_words / len(texts))
print("Average proportion of unique verb conjucations per text per level:", average_total_unique_verbforms / len(texts))
print("Average proportion of unique noun conjugations per text per level:", average_total_unique_nounforms / len(texts))
print("Average proportion of unique adjective conjugations per text per level:", average_total_unique_adjectiveforms / len(texts))