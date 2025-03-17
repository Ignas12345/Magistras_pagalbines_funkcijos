def create_labels_dict(sample_list, labels_df, slice_for_samples = slice(12), row_of_labels_df = 0):
  label_dict = {}
  #nes ten po meginio kodo dar prideda zodi "diagnosis"
  labels_df.columns = [col[:12] for col in labels_df.columns]
  for sample in sample_list:
    if sample[slice_for_samples] in labels_df.columns:
      label_dict[sample] = labels_df.loc[row_of_labels_df, sample[slice_for_samples]]
  return label_dict

def create_custom_labels(group_label_dict: dict, sample_ordering: np.ndarray | None = None, reference_labels_dict: dict | None = None, remaining_label: str | None = None):
  #sample ordering is array with each element being a string corresponding to sample id
  #here samples should be inputed as sample_ids, indices in sample ordering or as group corresponding to a label in reference labels.
  #reference_labels dict should be samples_labels_dict - i.e., samples are keys and labels are values. This type of dict should generally be used as input to functions.
  # group_label dict has sample groups as keys (each key/group should be a tuple) and labels to be assgined to that group as values.
  # remainig label can be used to set a label that will be assigned to otherwise unmetioned samples that will assign the remaining samples (the ones not mentioned explicitly as groups in group_label dict)

  new_sample_labels_dict = {}
  sample_groups = group_label_dict.keys()
  used_samples = []
  if type(list(group_label_dict.keys())[0]) != tuple:
    raise Exception('group_label_dict keys should be tuples, even if they contain single elements')

  if reference_labels_dict is not None:
    reference_labels_samples = list(reference_labels_dict.keys())
    reference_labels_dict = invert_label_dict(reference_labels_dict, original_keys='samples')
    reference_labels_list = list(reference_labels_dict.keys())
  else:
    print('no reference dictionary provided')
    reference_labels_list = []

  if sample_ordering is None:
    sample_ordering_not_provided = True
    print('sample_ordering not provided, using samples from reference dictionary')
    sample_ordering = reference_labels_samples
  else:
    sample_ordering_not_provided = False

  for group in sample_groups:
    samples = []
    for element in group:

      if type(element) == str and (element in reference_labels_list): #in this case element of a group is a label that will retrieve a set of samples from the reference_labels dict
        old_label = element
        samples += reference_labels_dict[old_label]

      elif type(element) == str and (element in sample_ordering): #in this case, we assume that sample_id is provided
        samples += [element]

      elif type(element) == int: #in this case retrieve sample_id according to sample ordering array
        if sample_ordering_not_provided:
          raise Exception('no sample order provided, but samples selected according index of an array')
        samples += [sample_ordering[element]]

      else:
        raise Warning('element ' + element + 'is neither a valid sample, nor a label in the provided arguments - (Check for misspellings or if reference_dictionary/sample_ordering are not provided)')


    for sample in samples:
      new_sample_labels_dict[sample] = group_label_dict[group]
    used_samples += samples

  if remaining_label is not None:
    if sample_ordering != []:
      unused_samples = list((set(sample_ordering)) - set(used_samples))
    else:
      unused_samples = list((set(reference_labels_samples)) - set(used_samples))
    for sample in unused_samples:
      new_sample_labels_dict[sample] = remaining_label

  return new_sample_labels_dict

def create_label_colors(label_list, color_list = None, default_list = ['blue', 'red', 'orange', 'cyan', 'purple', 'black', 'brown', 'yellow']):
  if color_list is None:
    color_list = default_list
  label_color_dict = {}
  for i in range(len(label_list)):
    label_color_dict[label_list[i]] = color_list[i]
  return label_color_dict

def plot_expressions_two_features(df_1, feature_1, feature_2, sample_ordering, df_2 = None, samples = None, label_dict = None, label_color_dict = None, extra_label_dict = None, xlim = None, ylim = None, show_legend = True, title = None):
  #paima df, kuriu rows yra observazijos ir padaro dvieju bruozu plot'a.

  if df_2 is not None:
    if np.any(df_2.index != sample_ordering):
      print('Warning: df_2 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

  print(np.any(df_1.index != sample_ordering))
  if np.any(df_1.index != sample_ordering):
    print('Warning: df_1 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

  #nuo cia assuminam, kad samples abiejose df yra isrikiuoti ppagal ta tvarka

  if df_2 is None:
    df_2 = df_1

  if label_dict is None:
    label_dict = {}
    for sample in sample_ordering:
      label_dict[sample] = 'sample'
  labels = list(set(label_dict.values()))

  if label_color_dict is None:
    label_color_dict = create_label_colors(labels)

  x_data = df_1[feature_1].to_numpy()
  y_data = df_2[feature_2].to_numpy()

  plt.figure()
  for label in labels:
    samples_with_label = np.where(np.array(list(label_dict.values())) == label)[0]
    plt.scatter(x_data[samples_with_label], y_data[samples_with_label], label = label, c = label_color_dict[label])
  if show_legend:
    plt.legend()
  plt.xlabel(feature_1)
  plt.ylabel(feature_2)
  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)
  if title is not None:
    plt.title(title)
  plt.show()

def perform_method(model, data_df: str | np.ndarray | None = None, supervised = False, sample_label_dict: dict | None = None, transpose_df=False, center_data: bool = True, scale_data: bool = False,
                              return_model=False, outliers_to_filter = None, use_original_feature_names = False, scale_back = False):
    """
    Performs dimensionality reduction using the provided model.
    The input data should have observations per row and features per column.
    Scale data can either be True/False. If the data is not centered, then it will also not be scaled.
    """
    if isinstance(data_df, str):
        print('Using array from URL - there may be errors if it is not the right format')
        data_df = pd.read_csv(data_df, index_col=0)
    if transpose_df:
        data_df = data_df.T
    feature_names = data_df.columns
    sample_ids = data_df.index
    data_array = data_df.to_numpy()

    if scale_data and center_data:
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)
    elif center_data:
        scaler = StandardScaler(with_std=False)
        data_array = scaler.fit_transform(data_array)
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

    if scale_back:
        transformed_data = scaler.inverse_transform(transformed_data)

    if use_original_feature_names:
        transformed_df = pd.DataFrame(data=transformed_data, columns=feature_names[model.support_], index=sample_ids)
    else:
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
