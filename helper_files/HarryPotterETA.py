import numpy as np
import pandas as pd
import re
import seaborn as sns
import nltk
import plotly_express as px
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import plotly_express as px
from gensim.models import word2vec
from gensim import corpora, models
from sklearn.manifold import TSNE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import pprint
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

import warnings
warnings.filterwarnings('ignore')


## IMPORTING

def importFiles(file_name):
    LINES = pd.DataFrame(open(file_name, 'r', encoding='utf-8-sig').readlines(), columns=['line_str'])
    LINES.index.name = 'line_num'
    LINES.line_str = LINES.line_str.str.replace(r'\n+', ' ', regex=True).str.strip()
    return LINES


## CHUNKING DATA

# function to find chapter chunks
def chunk_chaps(lines, chap_pat, book_num, OHCO):
    chap_lines = lines.line_str.str.match(chap_pat, case=True)
    
    # book specific regex
    if book_num == 1:
        a = lines.loc[chap_lines]
        a = a.drop([2394, 3124, 3127, 3147, 3182, 3196, 3197, 4990, 4992, 11155]) # 11155 because it's the second line of a chapter title - we don't want it to count as a standalone chapter title
        chap_titles = a.index.values.tolist()
    
    elif book_num == 2:
        # further specific regex required (remove lines ending with punctuation not indicative of chapter title)
        a = lines.loc[chap_lines]
        # removes lines with digits, ?, !, ., and -
        a = a[~a['line_str'].str.contains(r'[\d?!,.-]')]
        # removes lines that finish with the characters ", ., !, and I
        a = a[~a["line_str"].str.endswith(('”', '.', '!', 'I'))]
        a = a.drop([1019, 2851, 2855, 4266, 6147, 10864])
        chap_titles = a.index.values.tolist()
        
    elif book_num == 3:
        # chap_pat = r"^[^\S\n]*([A-Z][^a-z]*)((?:\n(?![^\S\n]*[A-Z][^a-z\n]*$).*)*)$"
        # further specific regex required (remove lines ending with punctuation not indicative of chapter title)
        a = lines.loc[chap_lines] # has too many observations, but we can further pare these down
        a = a[~a['line_str'].str.contains(r'[\d?!.-;]')] # removes lines with digits, ?, !, ., and -
        a = a[~a["line_str"].str.endswith(('”', '.', '!', 'I', 'EMPLOYEE', 'PRIZE', 'OWN', 'YOU', 'LILY'))] # removes lines ending with these things
        a = a.drop([1723, 2373, 9413, 9718, 15425, 17183, 17598]) # last 3 are secondary title lines - remove these later for book 3
        chap_titles = a.index.values.tolist()
        
    elif book_num == 4:
        # chap_pat = r"^[^\S\n]*([A-Z][^a-z]*)((?:\n(?![^\S\n]*[A-Z][^a-z\n]*$).*)*)$"
        # further specific regex required (remove lines ending with punctuation not indicative of chapter title)
        a = lines.loc[chap_lines] # has too many observations, but we can further pare these down
        a = a[~a['line_str'].str.contains(r'[\d?!;]')] # removes lines with digits, ?, !, ., and ;
        a = a[~a["line_str"].str.endswith(('”', '.', '!', 'I', 'WORLD', 'STINKS', 'MISTAKE', 'POTTER', 'RIDDLE'))] # removes lines ending with these things
        a = a.drop([11945, 10281]) # drop last lines that shouldn't be selected
        chap_titles = a.index.values.tolist()
        
    elif book_num == 5:
        # chap_pat = r"^[^\S\n]*([A-Z][^a-z]*)((?:\n(?![^\S\n]*[A-Z][^a-z\n]*$).*)*)$"
        # further specific regex required (remove lines ending with punctuation not indicative of chapter title)
        a = lines.loc[chap_lines] # has too many observations, but we can further pare these down
        a = a[~a['line_str'].str.contains(r'[\d?!;]')] # removes lines with digits, ?, !, and ;
        a = a[~a["line_str"].str.endswith(('”', '.', '!'))] # removes lines ending with these things
        a = a.drop([916, 1216, 1500, 1581, 1844, 1952, 3362, 3731, 3732, 3766, 3777, 3785, 4794, 4795, 5594, 6120, 6121, 6122, 6350, 10216, 10219, 10858, 10861, 11341, 12553, 16328, 17412, 17414,
                   19936, 19998, 20032, 23056, 23058, 23273, 23289, 23305, 23353, 23399, 23401, 23403, 23444, 26538, 27597, 27603, 27614, 27657, 30989, 30993, 31090, 31092, 31336, 31349, 32914,
                   32916, 32917, 32919, 33058, 33070, 35478, 35491, 36451, 36458, 37319, 37357, 37398, 43684, 43685, 46208, 46857, 48038]) # drop last lines that shouldn't be selected
        chap_titles = a.index.values.tolist()
        
    elif book_num == 6:
        # chap_pat = r"^[^\S\n]*([A-Z][^a-z]*)((?:\n(?![^\S\n]*[A-Z][^a-z\n]*$).*)*)$"
        # further specific regex required (remove lines ending with punctuation not indicative of chapter title)
        a = lines.loc[chap_lines] # has too many observations, but we can further pare these down
        a = a[~a['line_str'].str.contains(r'[\d?!.;]')] # removes lines with digits, ?, !, ., and ;
        a = a[~a["line_str"].str.endswith(('”', '.', '!'))] # removes lines ending with these things
        a = a.drop([1924, 1997, 1999, 2001, 5046, 5447, 5714, 5718, 5719, 5721, 17672, 27286]) # drop last lines that shouldn't be selected
        chap_titles = a.index.values.tolist()
        
    elif book_num == 7:
        # chap_pat = r"^[^\S\n]*([A-Z][^a-z]*)((?:\n(?![^\S\n]*[A-Z][^a-z\n]*$).*)*)$"
        # further specific regex required (remove lines ending with punctuation not indicative of chapter title)
        a = lines.loc[chap_lines] # has too many observations, but we can further pare these down
        a = a[~a['line_str'].str.contains(r'[\d?!.;]')] # removes lines with digits, ?, !, ., and ;
        a = a[~a["line_str"].str.endswith(('”', '.', '!'))] # removes lines ending with these things
        a = a.drop([766, 10376, 10378, 10877, 10878, 11262, 11263, 12346, 12487, 12532, 12534, 12538, 12540, 12611, 16481, 16484, 17597, 19978, 19984, 20328]) # drop last lines that shouldn't be selected
        chap_titles = a.index.values.tolist()
        
        
    # assign numbers to chapters
    lines.loc[chap_titles, 'chap_num'] = [i+1 for i in range(lines.loc[chap_titles].shape[0])]
    # forward fill chapter numbers
    lines.chap_num = lines.chap_num.ffill()
    # clean up
    lines = lines.drop(chap_titles)
    if book_num in {1, 3, 5, 7}: 
        if book_num == 1:
            addl_titles = [4215, 11155]
        if book_num == 3:
            addl_titles = [15425, 17183, 17598] # second title lines for 3 chapters in book 3
        if book_num == 5:
            addl_titles = [3362, 5594, 19936, 26538]
        if book_num == 7:
            addl_titles = [12346, 17597, 20328]
        lines = lines.drop(addl_titles) # drop additional title lines that were not caught before
    
    # Make big string for each chapter
    chaps = lines.groupby(OHCO[:1])\
        .line_str.apply(lambda x: '\n'.join(x))\
        .to_frame('chap_str')
    chaps['chap_str'] = chaps.chap_str.str.strip() # clip cruft from chap strings
    
    return a, chaps


def chap_to_par(chaps, para_pat, OHCO):
    paras = chaps['chap_str'].str.split(para_pat, expand=True).stack()\
        .to_frame('para_str').sort_index()
    paras.index.names = OHCO[:2]
    paras = paras['para_str'].str.replace("\n", " ").to_frame('para_str') # replace any lingering line break symbols that do not represent paragraph breaks
    
    return paras


def par_to_sent(paras, sent_pat, OHCO):
    # preprocessing - remove punctuation after 'Mr.' and 'Mrs.'
    paras = paras['para_str'].str.replace("Mrs[.]", "Mrs", regex=True).to_frame('para_str')
    paras = paras['para_str'].str.replace("Mr[.]", "Mr", regex=True).to_frame('para_str')
    
    # create sents dataframe
    sents = paras['para_str'].str.split(sent_pat, expand=True).stack()\
        .to_frame('sent_str')
    sents.index.names = OHCO[:3]
    
    # Remove empty paragraphs
    sents = sents[~sents['sent_str'].str.match(r'^\s*$')]
    
    # CRUCIAL TO REMOVE BLANK TOKENS
    sents.sent_str = sents.sent_str.str.strip()
    
    return sents


def sent_to_tok_pos(sents, tok_pat, OHCO):
    tokens = sents.sent_str\
            .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))\
            .stack()\
            .to_frame('pos_tuple')
    tokens.index.names = OHCO[:4]
    
    return tokens


## EXTRACT VOCAB

def extract_vocab(tokens):
    vocab = tokens.term_str.value_counts().to_frame('n')
    vocab.index.name = 'term_str'
    vocab['p'] = vocab.n / vocab.n.sum()
    vocab['i'] = -np.log2(vocab.p)
    vocab['n_chars'] = vocab.index.str.len()
    
    return vocab


## CREATE BOW

def create_bow(CORPUS, bag, item_type='term_str'):
    BOW = CORPUS.groupby(bag+[item_type])[item_type].count().to_frame('n')
    return BOW


## ANNOTATE VOCAB

def get_tfidf_dfidf(BOW, tf_method='max', df_method='standard', item_type='term_str'):    
    DTCM = BOW.n.unstack() # Create Doc-Term Count Matrix
    if tf_method == 'sum':
        TF = (DTCM.T / DTCM.T.sum()).T
    elif tf_method == 'max':
        TF = (DTCM.T / DTCM.T.max()).T
    elif tf_method == 'log':
        TF = (np.log2(DTCM.T + 1)).T
    elif tf_method == 'raw':
        TF = DTCM
    elif tf_method == 'bool':
        TF = DTCM.astype('bool').astype('int')
    else:
        raise ValueError(f"TF method {tf_method} not found.")
        
        
    DF = DTCM.count() # Assumes NULLs 
    N_docs = len(DTCM)
    
    if df_method == 'standard':
        IDF = np.log10(N_docs/DF) # This what the students were asked to use
    elif df_method == 'textbook':
        IDF = np.log10(N_docs/(DF + 1))
    elif df_method == 'sklearn':
        IDF = np.log10(N_docs/DF) + 1
    elif df_method == 'sklearn_smooth':
        IDF = np.log10((N_docs + 1)/(DF + 1)) + 1
    else:
        raise ValueError(f"DF method {df_method} not found.")
    
    TFIDF = TF * IDF
    
    DFIDF = DF * IDF
    
    TFIDF = TFIDF.fillna(0)

    return TFIDF, DFIDF


def get_book_from_corpus(CORPUS, book_id):
    BOOK = CORPUS.loc[book_id].copy()
    return BOOK


def get_TTM(BOOK):
    TTM = pd.get_dummies(BOOK['term_str'], columns=['term_str'], prefix='', prefix_sep='', drop_first=True)\
        .reset_index(drop=True).iloc[:,1:]
    TTM.index.name = 'time_id'
    return TTM


## LDA

def create_lda_model(DOCS, max_features, ngram_range, stop_words, n_components, max_iter, learning_offset, random_state, n_topics, n_top_terms):
    count_engine = CountVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    count_model = count_engine.fit_transform(DOCS.doc_str)
    TERMS = count_engine.get_feature_names_out()
    
    VOCAB = pd.DataFrame(index=TERMS)
    VOCAB.index.name = 'term_str'
    
    DTM = pd.DataFrame(count_model.toarray(), index=DOCS.index, columns=TERMS)
    
    VOCAB['doc_count'] = DTM.astype('bool').astype('int').sum()
    DOCS['term_count'] = DTM.sum(1)
                       
    lda_engine = LDA(n_components=n_components, max_iter=max_iter, learning_offset=learning_offset, random_state=random_state)
    TNAMES = [f"T{str(x).zfill(len(str(n_topics)))}" for x in range(n_topics)]
    
    lda_model = lda_engine.fit_transform(count_model)
    
    THETA = pd.DataFrame(lda_model, index=DOCS.index)
    THETA.columns.name = 'topic_id'
    THETA.columns = TNAMES
    
    PHI = pd.DataFrame(lda_engine.components_, columns=TERMS, index=TNAMES)
    PHI.index.name = 'topic_id'
    PHI.columns.name  = 'term_str'
    
    TOPICS = PHI.stack().to_frame('topic_weight').groupby('topic_id')\
    .apply(lambda x: x.sort_values('topic_weight', ascending=False)\
        .head(n_top_terms).reset_index().drop('topic_id', axis=1)['term_str'])
    
    return VOCAB, DOCS, TNAMES, THETA, PHI, TOPICS


## Word2Vec

def complete_analogy(A, B, C, model, n=2):
    try:
        cols = ['term', 'sim']
        return pd.DataFrame(model.wv.most_similar(positive=[B, C], negative=[A])[0:n], columns=cols)
    except KeyError as e:
        print('Error:', e)
        return None
    
    
def get_most_similar(positive, model, negative=None):
    return pd.DataFrame(model.wv.most_similar(positive, negative), columns=['term', 'sim'])


## SENTIMENT ANALYSIS

def plot_sentiments(df, emo='polarity'):
    FIG = dict(figsize=(25, 5), legend=True, fontsize=14, rot=45)
    df[emo].plot(**FIG)
    

## SEMANTIC SEARCH
    
def search_potter(concept, top_k, corpus_embeddings, CORPUS_by_para, books = [1, 2, 3, 4, 5, 6, 7]):
    # Get book indexes and filter corpus embeddings by book
    book_indexes = list(np.concatenate(([CORPUS_by_para[CORPUS_by_para['book_id'] == i].index for i in books])))
    
    corpus_embeddings_subset = corpus_embeddings[book_indexes]
    CORPUS_by_para_subset = CORPUS_by_para.iloc[book_indexes]
    
    #Encode the query
    query_embedding = model.encode(concept, convert_to_tensor=True)
    
    # Generate search hits
    potter_search_hits = util.semantic_search(query_embedding, corpus_embeddings_subset, top_k=top_k)[0]
    
    # Get row indexes of the top k hits
    potter_hit_indexes = [x['corpus_id'] for x in potter_search_hits]
    
    return pprint.pprint([{"Book Number: " + str(CORPUS_by_para_subset.iloc[x]["book_id"]) + 
             ", Chapter Number: " + 
             str(int(CORPUS_by_para_subset.iloc[x]["chap_num"]))
             :CORPUS_by_para_subset.iloc[x]["para_str"]} for x in potter_hit_indexes])