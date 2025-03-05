import numpy as np
import pandas as pd


def format_data_frame(df:str|pd.DataFrame, chars_slice = None,
                      sample_order = None, return_numpy = False, has_features = True,
                      return_feature_ids = True, return_sample_ids = False, sample_start_index = 1,
                      feature_id_index = 0, index_name = None, transpose_df = True):
  #Paima df, su obzervacijomis stulpeliuose. Sugrazina df su meginiais stulpeliuose ir bruozu pirmame stulpelyje - gali transposint. Array sutampa su df. Taip pat, davus meginiu tvarka, isrikiuoja meginius.

  message = ''

  if type(df) == str:
    print('reading df from url: ' + df)
    df = pd.read_csv(df)
  df = df.copy()

  if has_features == False:
    sample_start_index = 0
    return_feature_ids = False
    message += 'df has no features column\n'

  if chars_slice is not None:
    df.columns = [col[chars_slice] for col in df.head().columns]
    message += 'sample names truncated using: ' + str(chars_slice) + '\n'

  sample_ids = df.columns[sample_start_index:]

  if sample_order is not None:
    samples = df.loc[:, sample_order].copy()
    message += 'df reordered' + '\n'
  else:
    samples = df.iloc[:, sample_start_index:].copy()
  if has_features:
    df = pd.concat([df.iloc[:,feature_id_index], samples], axis=1)
  else:
    df = samples

  if index_name is not None:
    df.rename(columns = {df.columns[0]: index_name}, inplace = True)

  if has_features:
    feature_ids = df[df.columns[0]].to_numpy()
    array = df.drop(df.columns[0], axis = 1, inplace=False).to_numpy()
  else:
    array = df.to_numpy()

  if transpose_df:
    df = df.T.copy()
    if has_features:
      df.columns = df.iloc[0]
      df = df.drop(df.index[0])
    array = array.T
    print('df transposed')

  tuple_to_return = (df,)
  returns_for_print = '(df'

  if return_numpy:
   tuple_to_return = (df, array,)
   returns_for_print += ', array'

  if return_feature_ids:
    tuple_to_return = tuple_to_return + (feature_ids,)
    returns_for_print += ', feature_ids'

  if return_sample_ids:
    tuple_to_return = tuple_to_return + (sample_ids,)
    returns_for_print += ', sample_ids'

  returns_for_print += ')'
  message += 'final shape of df: ' + str(df.shape) + '\n'
  if return_numpy:
    message += 'final shape of array: ' + str(array.shape) + '\n'

  print(message)
  print('returning: ' + returns_for_print)

  return tuple_to_return

def prepare_sample_list(elements, sample_names:list = None, label_sample_dict:dict = None, sample_ordering: np.ndarray = None):
  '''checks elements against sample_names, a label dictionary and a sample ordering. Returns a list of sample names. At least one of
  sample_names, label_sample_dict or sample_ordering should be provided'''
  samples_to_return = []

  if sample_names is None:
    sample_names = []
  if label_sample_dict is None:
    label_sample_dict = {}

  for element in elements:
      if element in sample_names:
        samples_to_return.append(element)
      elif element in label_sample_dict:
        samples_to_return += label_sample_dict[element]
      elif type(element) == int:
        if sample_ordering is None:
          raise Exception('sample_ordering not provided, but sample inputted as a numerical index')
        samples_to_return.append(sample_ordering[element])
      else:
        raise Warning('sample ' + element + 'is neither a valid sample, nor a label in the provided arguments - (Check for misspellings or if valid dictionaries, namings are provided)')

  return samples_to_return  

def create_labels_from_df(labels_df: str | pd.DataFrame, sample_list: list | None = None, slice_for_samples = slice(12), feature_to_use = None, row_of_labels_df = 0, return_labels_samples = False, delete_empty_keys = False):
  labels_samples_dict = {}
  samples_labels_dict = {}

  if type(labels_df) == str: # labels_df can be provided as a url also (if it is passed as a string)
    labels_df = pd.read_csv(labels_df)

  if feature_to_use is None:
   index_to_use = labels_df.index[row_of_labels_df]
  else:
    index_to_use = feature_to_use
  #nes ten po meginio kodo dar prideda zodi "diagnosis"
  labels_df.columns = [col[:12] for col in labels_df.columns]
  unique_labels = list(set(labels_df.iloc[row_of_labels_df,:]))

  for label in unique_labels:
    labels_samples_dict[label] = []

  if sample_list is None:
    print('sample list not provided - labels created for patients in the label df, instead of samples')
    sample_list = labels_df.columns

  for sample in sample_list:
    if sample[slice_for_samples] in labels_df.columns:
      label = labels_df.loc[index_to_use, sample[slice_for_samples]]
      labels_samples_dict[label].append(sample)
      samples_labels_dict[sample] = label

  if delete_empty_keys:
    for key in list(labels_samples_dict.keys()):
      if len(labels_samples_dict[key]) == 0:
        del labels_samples_dict[key]

  if return_labels_samples:
    return samples_labels_dict, labels_samples_dict
  else:
    return samples_labels_dict

def invert_label_dict(dict_to_invert, original_keys: str):
  ''' originl_keys asks if keys of the original dict are 'samples' or 'labels' '''
  inverted_dict = {}

  #go from dict that has samples as keys to a dict that has labels as keys
  if original_keys == 'samples':
    for label in set(dict_to_invert.values()):
      inverted_dict[label] = []
    for sample in dict_to_invert.keys():
      inverted_dict[dict_to_invert[sample]].append(sample)

  #go from dict that has labels as keys to a dict that has values as keys
  if original_keys == 'labels':
    for label in dict_to_invert.keys():
      for sample in dict_to_invert[label]:
        inverted_dict[sample] = label

  return inverted_dict

def create_custom_labels(group_label_dict: dict, sample_ordering: np.ndarray | None = None, reference_labels_dict: dict | None = None, remaining_label: str | None = None, use_reference_for_remaining:bool = False):
  # sita funkcija skirta susikurti savam etikeciu zodnynui
  # sample ordering is array with each element being a string corresponding to sample id
  # here samples should be inputed as sample_ids, indices in sample ordering or as group corresponding to a label in reference labels.
  # reference_labels dict should be samples_labels_dict - i.e., samples are keys and labels are values. This type of dict should generally be used as input to functions.
  # group_label dict has sample groups as keys (each key/group should be a tuple) and labels to be assgined to that group as values.
  # remainig label can be used to set a label that will be assigned to otherwise unmetioned samples that will assign the remaining samples (the ones not mentioned explicitly as groups in group_label dict)
  # use use_reference_for_remaining is used to assign labels from reference dict to unmentioned samples (remaining label is ignored, reference_labels_dict must be passed)

  new_sample_labels_dict = {}
  label_sample_dict = None
  used_samples = []
  sample_groups = group_label_dict.keys()

  if sample_ordering is None:
    if reference_labels_dict is None:
      raise Exception('no reference dictionary provided, and no sample ordering provided')
    print('sample_ordering not provided, using samples from reference dictionary')
    sample_names = list(reference_labels_dict.keys())
  else:
    sample_names = sample_ordering

  if type(list(group_label_dict.keys())[0]) != tuple:
    raise Exception('group_label_dict keys should be tuples, even if they contain single elements')

  if reference_labels_dict is None:
    print('no reference dictionary provided')
    if use_reference_for_remaining:
      raise Exception('no reference dictionary provided, but use_reference_for_remaining is set to True')
  else:
    label_sample_dict = invert_label_dict(reference_labels_dict, original_keys='samples')


  for group in sample_groups:
    samples = prepare_sample_list(elements=group, sample_names = sample_names, label_sample_dict = label_sample_dict, sample_ordering=sample_ordering)

    for sample in samples:
      new_sample_labels_dict[sample] = group_label_dict[group]
    used_samples += samples


  unused_samples = list((set(sample_names)) - set(used_samples))
  if (remaining_label is not None) and (use_reference_for_remaining == False):
    for sample in unused_samples:
      new_sample_labels_dict[sample] = remaining_label
  elif use_reference_for_remaining:
    for sample in unused_samples:
      new_sample_labels_dict[sample] = reference_labels_dict[sample]

  return new_sample_labels_dict

def create_label_colors(labels:list|dict, color_list = None, default_list = ['blue', 'red', 'orange', 'cyan', 'purple', 'black', 'brown', 'yellow', 'gray']):
  if(type(labels) == dict):
    label_list = list(set(labels.values()))
  else:
    label_list = labels

  if color_list is None:
    color_list = default_list

  label_color_dict = {}
  for i in range(len(label_list)):
    label_color_dict[label_list[i]] = color_list[i]

  return label_color_dict  