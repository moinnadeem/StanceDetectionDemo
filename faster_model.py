from run_saved_model import *
import var

bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = load_vectorizers()
tokenizer = load_tokenizer()
embedding_matrix = load_embedding_matrix()

graph1 = tf.Graph()
graph2 = tf.Graph()

sess1 = tf.Session()
saver = tf.train.import_meta_graph(var.RELATED_UNRELATED_MODEL_PATH + ".meta")
saver.restore(sess1, tf.train.latest_checkpoint(var.RELATED_UNRELATED_MODEL_FOLDER))
graph1 = tf.get_default_graph()

sess2 = tf.Session(graph=graph2)
# Load 3 label model
# with tf.Session(graph=graph2) as sess2:
sess2.__enter__()
saver = tf.train.import_meta_graph(var.THREE_LABEL_MODEL_PATH + ".meta")
saver.restore(sess2, tf.train.latest_checkpoint(var.THREE_LABEL_MODEL_FOLDER))
graph2 = tf.get_default_graph()

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

    related_unrelated_preds = []
    related_unrelated_logits = []
    three_label_preds = []
    three_label_logits = []

    # Run related / unrelated model
           
    features_pl = graph1.get_tensor_by_name("features_pl:0")
    stances_pl = graph1.get_tensor_by_name("stances_pl:0")
    keep_prob_pl = graph1.get_tensor_by_name("keep_prob_pl:0")
    domains_pl = graph1.get_tensor_by_name("domains_pl:0")
    gr_pl = graph1.get_tensor_by_name("gr_pl:0")
    lr_pl = graph1.get_tensor_by_name("lr_pl:0")
    
    p_loss = graph1.get_tensor_by_name("p_loss:0")
    p_predict = graph1.get_tensor_by_name("p_predict:0")
    p_logits = graph1.get_tensor_by_name("Softmax:0")
    d_loss = graph1.get_tensor_by_name("d_loss:0")
    d_predict = graph1.get_tensor_by_name("d_predict:0")
    l2_loss = graph1.get_tensor_by_name("l2_loss:0")

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
            sess1.run([p_predict, d_predict, p_loss, p_logits, d_loss,
                      l2_loss], feed_dict=batch_feed_dict)

        related_unrelated_preds.extend(lpred)

        plogits = [[plogit[0], plogit[3]] for plogit in plogits]
        related_unrelated_logits.extend(plogits)

            
    if var.USE_TF_VECTORS:
        features_pl = graph2.get_tensor_by_name("features_pl:0")
    
    if var.USE_CNN_FEATURES:
        embedding_matrix_pl = graph2.get_tensor_by_name("embedding_matrix_pl:0")
        headline_words_pl = graph2.get_tensor_by_name("headline_words_pl:0")
        body_words_pl = graph2.get_tensor_by_name("body_words_pl:0")

    if var.ADD_FEATURES_TO_LABEL_PRED:
        p_features_pl = graph2.get_tensor_by_name("p_features_pl:0")

    stances_pl = graph2.get_tensor_by_name("stances_pl:0")
    keep_prob_pl = graph2.get_tensor_by_name("keep_prob_pl:0")
    domains_pl = graph2.get_tensor_by_name("domains_pl:0")
    gr_pl = graph2.get_tensor_by_name("gr_pl:0")
    lr_pl = graph2.get_tensor_by_name("lr_pl:0")
    
    p_loss = graph2.get_tensor_by_name("p_loss:0")
    p_predict = graph2.get_tensor_by_name("p_predict:0")
    p_logits = graph2.get_tensor_by_name("Softmax:0")
    d_loss = graph2.get_tensor_by_name("d_loss:0")
    d_predict = graph2.get_tensor_by_name("d_predict:0")
    l2_loss = graph2.get_tensor_by_name("l2_loss:0")

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
            sess2.run([p_predict, d_predict, p_loss, p_logits, d_loss,
                      l2_loss], feed_dict=batch_feed_dict)

        three_label_preds.extend(lpred)

        plogits = [[plogit[0], plogit[1], plogit[2]] for plogit in plogits]
        three_label_logits.extend(plogits)
        
    related_unrelated_preds = ["Unrelated" if related_unrelated_preds[i] == 3 else "Related" for i in range(len(related_unrelated_preds))]
    three_label_preds = ["Agree" if three_label_preds[i] == 0 else "Disagree" if three_label_preds[i] == 1 else "Discuss" for i in range(len(three_label_preds))]

    return claims, documents, related_unrelated_preds, related_unrelated_logits, three_label_preds, three_label_logits


def query(claim, document):
    claim_doc_feat_vectors, claim_seqs_padded, document_seqs_padded, claims, documents = process_input(claim, document, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, tokenizer)
    response = run_model(claim_doc_feat_vectors, claim_seqs_padded, document_seqs_padded, embedding_matrix, claims, documents)
    return response
