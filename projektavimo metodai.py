def perform_method(model, data_df: str | np.ndarray | None = None, supervised = False, sample_label_dict: dict | None = None, data_array: np.ndarray | None = None,
                              sample_ids=None, feature_ids=None, transpose_df=False, scale_data: bool = False, with_std = True,
                              return_model=False, outliers_to_filter = None):
    """
    Performs dimensionality reduction using the provided model.
    The input data should have observations per row and features per column.
    Scale data can either be True/False (if True Standard scaler is used) or a function that scales the data, such as mean subtraction - the function always
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

    if scale_data:
        data_array = StandardScaler(with_std=with_std).fit_transform(data_array)

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

def filter_samples_from_df(df, samples_to_filter, sample_ordering = None):
  if samples_to_filter is None:
    return df
  sample_names = df.index
  samples_to_filter = prepare_sample_list(elements = samples_to_filter, sample_names = sample_names, sample_ordering=sample_ordering)
  return df.drop(samples_to_filter)

def check_labels(sample_list, sample_label_dict, sample_ordering = None):
  for i, sample in enumerate(prepare_sample_list(elements = sample_list, sample_ordering=sample_ordering)):
    if sample_ordering is not None:
      print(str(np.where(sample_ordering == sample)[0]) +', '+sample +': '+labels_1[sample])
    else:
      print(sample +': '+sample_label_dict[sample])

