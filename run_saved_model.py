import numpy as np
import tensorflow as tf
import var
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from util import print_model_results, get_prediction_accuracies, get_composite_score, \
        get_f1_scores, save_predictions, get_feature_vectors, remove_stop_words, \
        get_body_sentences

def load_vectorizers():
    '''
    Load the BOW, TFREQ, and TFIDF vectorizers from pickle files.
    '''
    bow_vectorizer = pickle.load(open(var.PICKLE_SAVE_FOLDER + "bow_vectorizer.pickle", "rb")) 
    tfreq_vectorizer = pickle.load(open(var.PICKLE_SAVE_FOLDER + "tfreq_vectorizer.pickle", "rb"))
    tfidf_vectorizer = pickle.load(open(var.PICKLE_SAVE_FOLDER + "tfidf_vectorizer.pickle", "rb"))
    
    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def load_tokenizer():
    '''
    Load the word index, which maps numbers to words, and initialize a tokenizer with
    that word index. This function takes advantage of the fact that the tokenizer will
    assign words to numbers based on the order in which it sees the words.
    '''
    word_index = np.load(var.PICKLE_SAVE_FOLDER + "word_index.npy").item()

    # initialize tokenizer from word_index
    words = [(word, word_index[word]) for word in word_index]
    sorted(words, key=lambda x: x[1])
    vocab = " ".join([word[0] for word in words])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([vocab])

    return tokenizer


def process_input(claim, document, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, tokenizer):
    '''
    Given claim text and document text, prepare the input so that it can be passed into the model.
    In particular, TF/TFIDF features are extracted and the inputs are also converted into arrays of word embeddings
    corresponding to the original claim and document.
    '''
    # Parse document into sentences
    document_sentences = get_body_sentences([document])[0]
    documents = [document] + document_sentences

    # Create corresponding claims
    claims = [claim for _ in range(len(document_sentences) + 1)]

    # Get TF/TFIDF feature vectors
    claim_doc_feat_vectors = get_feature_vectors(claims, documents, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    claims_no_stop = remove_stop_words(claims)
    documents_no_stop = remove_stop_words(documents)

    claim_seqs = tokenizer.texts_to_sequences(claims_no_stop)
    document_seqs = tokenizer.texts_to_sequences(documents_no_stop)

    claim_seqs_padded = pad_sequences(claim_seqs, maxlen=var.CNN_HEADLINE_LENGTH)
    document_seqs_padded = pad_sequences(document_seqs, maxlen=var.CNN_BODY_LENGTH)

    return claim_doc_feat_vectors, claim_seqs_padded, document_seqs_padded, claims, documents


def load_embedding_matrix():
    '''
    Loads the embedding matrix from the pretrained model.
    '''
    return np.load(var.PICKLE_SAVE_FOLDER + "embedding_matrix.npy")


def run_model(claim_doc_feat_vectors, claim_seqs_padded, document_seqs_padded, embedding_matrix, claims, documents):
    '''
    Loads the pretrained tensorflow modeland passes in the desired inputs. The label predictions and logits
    associated with those predictions are returned in the response.
    '''
    num_data = len(claim_doc_feat_vectors)

    # dummy stances and domains for model
    stances = [0 for _ in range(num_data)]
    domains = [0 for _ in range(num_data)]
    indicies = [i for i in range(num_data)]

    graph1 = tf.Graph()
    graph2 = tf.Graph()

    related_unrelated_preds = []
    related_unrelated_logits = []
    three_label_preds = []
    three_label_logits = []

    # Run related / unrelated model
    with tf.Session(graph=graph1) as sess:
        saver = tf.train.import_meta_graph(var.RELATED_UNRELATED_MODEL_PATH + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(var.RELATED_UNRELATED_MODEL_FOLDER))

        graph = tf.get_default_graph()
           
        features_pl = graph.get_tensor_by_name("features_pl:0")
        stances_pl = graph.get_tensor_by_name("stances_pl:0")
        keep_prob_pl = graph.get_tensor_by_name("keep_prob_pl:0")
        domains_pl = graph.get_tensor_by_name("domains_pl:0")
        gr_pl = graph.get_tensor_by_name("gr_pl:0")
        lr_pl = graph.get_tensor_by_name("lr_pl:0")
        
        p_loss = graph.get_tensor_by_name("p_loss:0")
        p_predict = graph.get_tensor_by_name("p_predict:0")
        p_logits = graph.get_tensor_by_name("Softmax:0")
        d_loss = graph.get_tensor_by_name("d_loss:0")
        d_predict = graph.get_tensor_by_name("d_predict:0")
        l2_loss = graph.get_tensor_by_name("l2_loss:0")

        for i in range(len(documents) // var.BATCH_SIZE + 1):
            batch_indices = indicies[i * var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
            batch_stances = [stances[i] for i in batch_indices]
            batch_features = [claim_doc_feat_vectors[i] for i in batch_indices]
            
            batch_feed_dict = {
                stances_pl: batch_stances,
                features_pl: batch_features,
                keep_prob_pl: 1.0
            }

            # Record loss and accuracy information for test
            lpred, dpred, ploss, plogits, dloss, l2loss = \
                sess.run([p_predict, d_predict, p_loss, p_logits, d_loss,
                          l2_loss], feed_dict=batch_feed_dict)

            related_unrelated_preds.extend(lpred)

            plogits = [[plogit[0], plogit[3]] for plogit in plogits]
            related_unrelated_logits.extend(plogits)

    with tf.Session(graph=graph2) as sess:
        # Load 3 label model
        saver = tf.train.import_meta_graph(var.THREE_LABEL_MODEL_PATH + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(var.THREE_LABEL_MODEL_FOLDER))

        graph = tf.get_default_graph()
        
        if var.USE_TF_VECTORS:
            features_pl = graph.get_tensor_by_name("features_pl:0")
        
        if var.USE_CNN_FEATURES:
            embedding_matrix_pl = graph.get_tensor_by_name("embedding_matrix_pl:0")
            headline_words_pl = graph.get_tensor_by_name("headline_words_pl:0")
            body_words_pl = graph.get_tensor_by_name("body_words_pl:0")

        if var.ADD_FEATURES_TO_LABEL_PRED:
            p_features_pl = graph.get_tensor_by_name("p_features_pl:0")

        stances_pl = graph.get_tensor_by_name("stances_pl:0")
        keep_prob_pl = graph.get_tensor_by_name("keep_prob_pl:0")
        domains_pl = graph.get_tensor_by_name("domains_pl:0")
        gr_pl = graph.get_tensor_by_name("gr_pl:0")
        lr_pl = graph.get_tensor_by_name("lr_pl:0")
        
        p_loss = graph.get_tensor_by_name("p_loss:0")
        p_predict = graph.get_tensor_by_name("p_predict:0")
        p_logits = graph.get_tensor_by_name("Softmax:0")
        d_loss = graph.get_tensor_by_name("d_loss:0")
        d_predict = graph.get_tensor_by_name("d_predict:0")
        l2_loss = graph.get_tensor_by_name("l2_loss:0")

        for i in range(len(documents) // var.BATCH_SIZE + 1):
            batch_indices = indicies[i * var.BATCH_SIZE: (i + 1) * var.BATCH_SIZE]
            batch_stances = [stances[i] for i in batch_indices]
            batch_domains = [domains[i] for i in batch_indices]
            batch_features = [claim_doc_feat_vectors[i] for i in batch_indices]
            
            batch_feed_dict = {stances_pl: batch_stances,
                               keep_prob_pl: 1.0}

            if var.USE_DOMAINS:
                batch_feed_dict[gr_pl] = 1.0
                batch_feed_dict[domains_pl] = batch_domains

            if var.USE_TF_VECTORS or var.ADD_FEATURES_TO_LABEL_PRED:
                if var.USE_TF_VECTORS:
                    batch_feed_dict[features_pl] = batch_features
                if var.ADD_FEATURES_TO_LABEL_PRED:
                    batch_feed_dict[p_features_pl] = batch_features

            if var.USE_CNN_FEATURES:
                batch_x_headlines = [claim_seqs_padded[i] for i in batch_indices]
                batch_x_bodies = [document_seqs_padded[i] for i in batch_indices]
                batch_feed_dict[headline_words_pl] = batch_x_headlines
                batch_feed_dict[body_words_pl] = batch_x_bodies
                batch_feed_dict[embedding_matrix_pl] = embedding_matrix

            # Record loss and accuracy information for test
            lpred, dpred, ploss, plogits, dloss, l2loss = \
                sess.run([p_predict, d_predict, p_loss, p_logits, d_loss,
                          l2_loss], feed_dict=batch_feed_dict)

            three_label_preds.extend(lpred)

            plogits = [[plogit[0], plogit[1], plogit[2]] for plogit in plogits]
            three_label_logits.extend(plogits)
        
    related_unrelated_preds = ["Unrelated" if related_unrelated_preds[i] == 3 else "Related" for i in range(len(related_unrelated_preds))]
    three_label_preds = ["Agree" if three_label_preds[i] == 0 else "Disagree" if three_label_preds[i] == 1 else "Discuss" for i in range(len(three_label_preds))]

    return claims, documents, related_unrelated_preds, related_unrelated_logits, three_label_preds, three_label_logits


def run_model_main(claim, document):
    '''
    Function which encapsulates the complete process of taking an input claim and document and retrieving relevant
    predictions from a pretrained model. Uses all helper functions defined above to complete the task.
    '''
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = load_vectorizers()
    tokenizer = load_tokenizer()
    embedding_matrix = load_embedding_matrix()
    claim_doc_feat_vectors, claim_seqs_padded, document_seqs_padded, claims, documents = process_input(claim, document, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, tokenizer)
    response = run_model(claim_doc_feat_vectors, claim_seqs_padded, document_seqs_padded, embedding_matrix, claims, documents)
    return response


