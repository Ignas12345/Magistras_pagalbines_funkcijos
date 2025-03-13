import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def perform_method(model, data_df: str | np.ndarray | None = None, supervised = False, sample_label_dict: dict | None = None, data_array: np.ndarray | None = None,
                              sample_ids=None, feature_ids=None, transpose_df=False, center_data: bool = True, scale_data: bool = False,
                              return_model=False, outliers_to_filter = None):
    """
    Performs dimensionality reduction using the provided model.
    The input data should have observations per row and features per column.
    Scale data can either be True/False. If the data is not centered, then it will also not be scaled.
    """
    if data_df is not None:
        if isinstance(data_df, str):
            print('Using array from URL - there may be errors if it is not the right format')
            data_df = pd.read_csv(data_df, index_col=0)
        if transpose_df:
            data_df = data_df.T
        if data_array is None:
            data_array = data_df.to_numpy()
        feature_ids = data_df.columns
        sample_ids = data_df.index

    if scale_data and center_data:
        data_array = StandardScaler().fit_transform(data_array)
    elif center_data:
        data_array = StandardScaler(with_std=False).fit_transform(data_array)
    elif scale_data:
        raise ValueError("data should not be scaled, unless first centered (set center_data to True)")

    if outliers_to_filter is not None:
        mask = [sample not in outliers_to_filter for sample in sample_ids]
        data_for_training = data_array[mask]
        sample_ids_for_training = sample_ids[mask]
        print('filtered samples for training phase: ' + str(outliers_to_filter))
    else:
        data_for_training = data_array
        sample_ids_for_training = sample_ids

    if supervised:
        if sample_label_dict is None:
            raise ValueError("Supervised models require a label dictionary.")
        labels = np.array([sample_label_dict[sample] for sample in sample_ids_for_training])

        model.fit(data_for_training, labels)
    else:
        model.fit(data_for_training)

    transformed_data = model.transform(data_array)
    transformed_df = pd.DataFrame(data=transformed_data, columns=[f'Component {i+1}' for i in range(transformed_data.shape[1])], index=sample_ids)

    tuple_to_return = (transformed_df,)
    message_return_tuple = f'Returning tuple: (transformed_data_df'

    if return_model:
        tuple_to_return += (model,)
        message_return_tuple += ', model'

    if len(tuple_to_return) == 1:
      print('Returning transformed data')
      return tuple_to_return[0]

    print(message_return_tuple + ')')
    return tuple_to_return


