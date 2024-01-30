import pandas as pd
import numpy as np
import torch
import pos
from nltk import word_tokenize
from reynir import Greynir
from tokenizer import split_into_sentences
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cefrdata.csv')

# Can be uncommented if the data is to be concatinated to the accompanying IGC data
# df2 = pd.read_csv('IGC_texts.csv')
# df = pd.concat([df,df2])

texts = df['text']

def get_features():

    cleantexts = []
    concatinated_vectors = []
    featurevectors = []

    device = torch.device("cuda")
    tagger = pos.Tagger = torch.hub.load( # The POS package is based on ABLTagger (see https://github.com/cadia-lvl/POS)
        repo_or_dir="cadia-lvl/POS",
        model="tag",
        device=device,
        force_reload=False,
        force_download=False,
    )

    # Used only if the texts should be embedded by IceBERT

    #tokenizer = AutoTokenizer.from_pretrained("mideind/IceBERT")
    #model = AutoModelForMaskedLM.from_pretrained("mideind/IceBERT")

    count = 1
    for i in texts:
        text = word_tokenize(i) # Tokenizing the text to keep track of total number of words per text
        #if len(text) > 1000: # Only to be used if the text length needs to be limited due to time constraints
        #    text = text[:1001]
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

        featurevector = []

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
        
        try:
            tags = tagger.tag_sent(text) # Getting the POS tags for each text
        except AssertionError: # Avoiding an error from POS saying that max_length is wrong
            continue
        tagged = list(zip(text, tags)) # Matching each word with the corresponding tag
        longwordscore = 0 # Keeping track of how many words exceed 6 letters

        cleantext = ""

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

            if tag.startswith("n") and tag.endswith("s"): # Excluding proper names,
                continue
            elif tag.startswith("t") and tag.endswith("a"): # years and indeclinable numbers,
                continue
            elif tag.startswith("t") and tag.endswith("p"): # percentages,
                continue
            elif tag.startswith("e") or tag.startswith("x") or tag.startswith("v") or tag.startswith("m"): # foreign words, unspecified words, emails/websites and symbols
                continue
            else:
                if len(word) > 6:
                    longwordscore += 1
                cleantext = cleantext + word + " "

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
            
        # the features that are commented out don't seem to improve the results here but can be uncommented for further experimentation
                    
        #featurevector.append(len(set(text)) / len(text)) # proportion of unique word forms in the text
        #featurevector.append(longwordscore / len(text))
        featurevector.append(total_adjectives / len(text))
        #featurevector.append(total_nouns / len(text))
        featurevector.append(total_verbs / len(text))
        #featurevector.append(total_pronouns / len(text))
        #featurevector.append(total_interrogatives / len(text))
        featurevector.append(total_particles / len(text))
        
        if anysent != 0: # Avoiding division by zero in case no sentence was parsed
            featurevector.append(subordinates / anysent)
            #featurevector.append(comp_subordinates / anysent)
            featurevector.append(que_subordinates / anysent)
            featurevector.append(rel_subordinates / anysent)
            featurevector.append(temp_subordinates / anysent)
            #featurevector.append(purp_subordinates / anysent)
            #featurevector.append(aknow_subordinates / anysent)
            #featurevector.append(cons_subordinates / anysent)
            #featurevector.append(caus_subordinates / anysent)
            #featurevector.append(cond_subordinates / anysent)
            #featurevector.append(compar_subordinates / anysent)
        else:
            featurevector.append(0)
            #featurevector.append(0)
            featurevector.append(0)
            featurevector.append(0)
            featurevector.append(0)
            #featurevector.append(0)
            #featurevector.append(0)
            #featurevector.append(0)
            #featurevector.append(0)
            #featurevector.append(0)
            #featurevector.append(0)

        if total_verbs != 0: # In all cases, avoiding division by zero in case no words of the corresponding POS was counted
            featurevector.append(subjunctive / total_verbs)
            featurevector.append(middle / total_verbs)
            #featurevector.append(passive / total_verbs)
            featurevector.append(past_verb / total_verbs)
            #featurevector.append(len(unique_verbforms) / total_verbs)
        else:
            featurevector.append(0)
            featurevector.append(0)
            #featurevector.append(0)
            featurevector.append(0)
            #featurevector.append(0)

        if total_adjectives != 0:
            featurevector.append(adj_superlative_oblique / total_adjectives)
            featurevector.append(superlative / total_adjectives)
            featurevector.append(len(unique_adjectiveforms) / total_adjectives)
        else:
            featurevector.append(0)
            featurevector.append(0)
            featurevector.append(0)

        if total_nouns != 0:
            featurevector.append(noun_definite_oblique / total_nouns)
            featurevector.append(len(unique_nounforms) / total_nouns)
        else:
            featurevector.append(0)
            featurevector.append(0)

        if total_interrogatives != 0:
            featurevector.append(interrogatives_oblique / total_interrogatives)
        else:
            featurevector.append(0)

        if total_pronouns != 0:
            featurevector.append(pronoun_1p / total_pronouns)
        else:
            featurevector.append(0)

        featurevectors.append(featurevector)
        count += 1

    # If only handpicked features should be used 
    return featurevectors

        # cleantexts.append(cleantext)

        # The following block of code concatinates the feature vectors and BERT embeddings of the text examples

        # encoded_text = tokenizer.encode(cleantext, return_tensors='pt', max_length=10000, truncation=True, padding='max_length')
        # list_tensor = torch.tensor(featurevector)
        # featuretensor = list_tensor.view(1, -1)
        # concatenated = torch.cat((featuretensor, encoded_text), dim=1).squeeze()

        # concatinated_vectors.append(concatenated.tolist())        

    # If only feature vectors and embeddings should be used
    #return concatinated_vectors

    # The following block of code concatinates the vectors and tf-idf vectors of the text examples

    # vectorizer1 = TfidfVectorizer()
    # tfidf_vectors = vectorizer1.fit_transform(cleantexts)
    # tokens = vectorizer1.get_feature_names_out()
    # df_tfidfvect = pd.DataFrame(data = tfidf_vectors.toarray(), index = [i for i in range(78)], columns = tokens)
    # concatenated2 = df_tfidfvect.apply(lambda row: concatenate_row(row, featurevectors), axis=1)
    # concatinated_final = concatenated2.values.tolist()

    # If tf-idf scores should be used along with the feature vectors (note that the concatination here must be changed in a small way to include both feature vectors and embeddings along with tf-idfs)
    #return concatinated_final

# This is a helper function for the previous block of code
def concatenate_row(row, lists):
    return row.tolist() + lists[row.name]

# This is a function that trains the gradient boosting model and measures its accuracy using 10-fold cross-validation or alternatively, measures accuracy of one test run and creates a confusion set
def train_model():
    X = get_features()
    y = df['cefrlevel'].astype('category').cat.codes.to_numpy()

    # Split the data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    scaler = StandardScaler()

    # Fit and transform the training data
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.fit_transform(X)

    # Model Training
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)#.fit(X_train_scaled, y_train)

    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    scores = cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=cv, n_jobs=-1)

    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))

    # y_pred = clf.predict(X_test_scaled)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    # cnf_matrix = confusion_matrix(y_test, y_pred)

    # class_names=["A1", "A2", "B1", "B2", "C1", "C2"] # name  of classes
    # fig, ax = plt.subplots()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names)
    # plt.yticks(tick_marks, class_names)
    # # create heatmap
    # sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    # ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    # plt.title('Confusion matrix', y=1.1)
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.show()
    
train_model()