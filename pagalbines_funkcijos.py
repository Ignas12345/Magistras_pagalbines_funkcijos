import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


'''Funkcijų sąrašas:
format_data_frame, prepare_sample_list, create_labels_from_df, create_custom_labels, invert_label_dict, create_label_colors,
plot_two_features, plot_two_features_use_text'''

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
  '''labels should be list or sample_label_dict'''
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

def plot_two_features(df_1, feature_1, feature_2, df_2 = None, samples_to_use:list|None = None, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None, samples_to_higlight: list|None = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, slice_for_samples:slice = None, use_numbers:bool = None):
  #paima df, kuriu rows yra observazijos ir padaro dvieju bruozu plot'a.
  #label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values
  #slice_for_samples, use_numbers argumentai yra naudojami kitame funckijos variante - ten kur su tekstu plotinna - cia jie nereikalingi

  '''
  if sample_ordering is not None:
    if df_2 is not None:
      if np.any(df_2.index != sample_ordering):
        print('Warning: df_2 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

    if np.any(df_1.index != sample_ordering):
      print('Warning: df_1 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

  #nuo cia assuminam, kad samples abiejose df yra isrikiuoti pagal ta tvarka arba kad meginiu pavadinimai naudojami
  '''
  df_1 = df_1.copy()
  if df_2 is None:
    df_2 = df_1.copy()
  else:
    df_2 = df_2.copy()

  if samples_to_use is None:
    samples_to_use = df_1.index.copy()
    use_all_samples = True
  else:
    use_all_samples = False

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in samples_to_use:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if use_all_samples == False:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df_1.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  '''
  x_data = df_1[feature_1].to_numpy()
  y_data = df_2[feature_2].to_numpy()
  '''
  x_data = df_1[feature_1]
  y_data = df_2[feature_2]

  plt.figure()

  if use_all_samples:
    for label in labels:
      samples_with_label = label_sample_dict[label]
      #plt.scatter(x_data[samples_with_label], y_data[samples_with_label], label = label, c = label_color_dict[label])
      plt.scatter(x_data.loc[samples_with_label], y_data.loc[samples_with_label], label = label, c = label_color_dict[label])
  else:
    for label in labels:
      samples_with_label = list(set(label_sample_dict[label]).intersection(set(samples_to_use)))
      if len(samples_with_label) > 0:
        #plt.scatter(x_data[samples_with_label], y_data[samples_with_label], label = label, c = label_color_dict[label])
        plt.scatter(x_data.loc[samples_with_label], y_data.loc[samples_with_label], label = label, c = label_color_dict[label])

  if samples_to_higlight is not None:
    samples_to_higlight = prepare_sample_list(elements = samples_to_higlight, sample_names = samples_to_use, label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)
    plt.scatter(x_data.loc[samples_to_higlight], y_data.loc[samples_to_higlight], label = 'highlighted', c = 'gold', marker='v')



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

def plot_two_features_use_text(df_1, feature_1, feature_2, df_2 = None, use_numbers = False, slice_for_samples:slice = None, samples_to_use:list|None = None, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None,
                                  xlim = None, ylim = None, show_legend = True, title = None):
  '''kaip kita plottinim'o funkcija, bet naudoja teksta arba skaicius vietoj tasku.'''
  #paima df, kuriu rows yra observazijos ir padaro dvieju bruozu plot'a.
  #label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values

  '''
  if sample_ordering is not None:
    if df_2 is not None:
      if np.any(df_2.index != sample_ordering):
        print('Warning: df_2 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

    if np.any(df_1.index != sample_ordering):
      print('Warning: df_1 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

  #nuo cia assuminam, kad samples abiejose df yra isrikiuoti pagal ta tvarka arba kad meginiu pavadinimai naudojami
  '''
  df_1 = df_1.copy()
  if df_2 is None:
    df_2 = df_1.copy()
  else:
    df_2 = df_2.copy()

  if use_numbers and sample_ordering is None:
    raise Exception('sample_ordering should be provided if using numbers to plot')

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in df_1.columns:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if samples_to_use is None:
    samples_to_use = df_1.index.copy()
  else:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df_1.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  x_data = df_1[feature_1].copy()
  y_data = df_2[feature_2].copy()

  plt.figure()
  used_labels = []
  patches_for_legend = []

  if xlim is None:
    x_min = min(x_data)
    x_max = max(x_data)
    print('feature "' +feature_1+ '" ranges from: ' + str(x_min) + ' to ' +str(x_max))
    buffer = (x_max - x_min)/10
    xlim = [x_min - buffer, x_max + buffer]
  if ylim is None:
    y_min = min(y_data)
    y_max = max(y_data)
    print('feature "' +feature_2+ '" ranges from: ' + str(y_min) + ' to ' +str(y_max))
    buffer = (y_max - y_min)/10
    ylim = [y_min - buffer, y_max + buffer]

  for label in labels:
    #used_x = np.array([])
    #used_y = np.array([])
    samples_with_label = list(set(label_sample_dict[label]).intersection(set(samples_to_use)))
    if len(samples_with_label) > 0:
      used_labels += [label]
      x_coords = x_data.loc[samples_with_label].to_numpy()
      y_coords = y_data.loc[samples_with_label].to_numpy()
      #used_x = np.append(used_x, x_coords)
      #used_y = np.append(used_y, y_coords)
      if use_numbers:
        indices = []
        for sample in samples_with_label:
          indices += list(np.where(sample_ordering == sample)[0])
        samples_with_label = indices
      elif slice_for_samples is not None:
        samples_with_label = [sample[slice_for_samples] for sample in samples_with_label]
      for x, y, t in zip(x_coords, y_coords, samples_with_label):
        if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:
           plt.text(x, y, t, fontsize=10, color= label_color_dict[label])


  for label in used_labels:
    patches_for_legend += [mpatches.Patch(color=label_color_dict[label], label=label)]



  plt.xlim(xlim)
  plt.ylim(ylim)
  if show_legend:
    plt.legend(handles = patches_for_legend)
  plt.xlabel(feature_1)
  plt.ylabel(feature_2)
  if title is not None:
    plt.title(title)

  plt.show()
