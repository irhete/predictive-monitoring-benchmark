import EncoderFactory

from bucketers.ClusterBasedBucketer import ClusterBasedBucketer
from bucketers.StateBasedBucketer import StateBasedBucketer
from bucketers.PrefixLengthBucketer import PrefixLengthBucketer
from bucketers.NoBucketer import NoBucketer
from bucketers.KNNBucketer import KNNBucketer
from sklearn.cluster import KMeans

        
def get_bucketer(method, encoding_method=None, case_id_col=None, cat_cols=None, num_cols=None, n_clusters=None, random_state=None, n_neighbors=None):

    if method == "cluster":
        bucket_encoder = EncoderFactory.get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        clustering = KMeans(n_clusters, random_state=random_state)
        return ClusterBasedBucketer(encoder=bucket_encoder, clustering=clustering)
        
    elif method == "state":
        bucket_encoder = EncoderFactory.get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        return StateBasedBucketer(encoder=bucket_encoder)
            
    elif method == "single":
        return NoBucketer(case_id_col=case_id_col)

    elif method == "prefix":
        return PrefixLengthBucketer(case_id_col=case_id_col)
    
    elif method == "knn":
        bucket_encoder = EncoderFactory.get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        return KNNBucketer(encoder=bucket_encoder, n_neighbors=n_neighbors)

    else:
        print("Invalid bucketer type")
        return None