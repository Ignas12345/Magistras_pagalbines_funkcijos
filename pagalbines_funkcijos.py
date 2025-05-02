import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


'''Funkcijų sąrašas:
format_data_frame, collapse_columns_by_string, filter_samples, filter_features, prepare_sample_list, shuffle_columns, calc_corr_for_sing_feat,
check_labels, create_labels_from_df, invert_label_dict, create_custom_labels, prepare_labels, create_label_colors,
plot_two_features, plot_two_features_use_text, plot_single_feature, plot_single_feature_use_text, perform_method,
read_feature_ranking_df, write_feature_ranking_df'''

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

def collapse_columns_by_string(df, string_slice=slice(-1, -12, -1)):
    """
    Collapses columns in a DataFrame based on matching string of column names (last `string_len` characters).

    Args:
        df (pd.DataFrame): DataFrame with features as columns and samples as rows.
        

    Returns:
        pd.DataFrame: DataFrame with collapsed columns (features), summed across matching key_strings.
    """

    key_strings = df.columns.to_series().str[string_slice]
    # Group column names by their string
    grouped = {}
    for string in key_strings.unique():
        cols = key_strings[key_strings == string].index.tolist()
        grouped[string] = cols

    # Build new DataFrame with collapsed columns
    collapsed_data = {}
    for string, cols in grouped.items():
        # Sum across the columns that match this string
        collapsed_data[cols[0]] = df[cols].sum(axis=1)

    # Construct new DataFrame
    collapsed_df = pd.DataFrame(collapsed_data)

    return collapsed_df

def filter_samples(df, samples_to_filter):
    mask = [sample not in samples_to_filter for sample in df.index]
    return df[mask].copy()

def filter_features(df, features_to_filter):
    mask = [feature not in features_to_filter for feature in df.columns]
    return df.loc[:, mask].copy()

def feature_filtering_by_class_means(df_to_use, class1_samples, class2_samples, class1_name = '1', class2_name = '2', threshold_to_keep = 45):
    '''paprasta funkcija, kuri atlieka bruozu filtravima pagal meginius ir ju klases.'''
    # Extract expression data for the two classs
    data1 = df_to_use.loc[class1_samples].copy()
    data2 = df_to_use.loc[class2_samples].copy()

    #By default leave only the features that reach 50 (45 for safety) RPM in any of the classes
    features_to_keep_1 = [feature for feature in data1.columns if data1[feature].mean() >= threshold_to_keep]
    features_to_keep_2 = [feature for feature in data2.columns if data2[feature].mean() >= threshold_to_keep]
    features_to_keep = list(set(features_to_keep_1).union(set(features_to_keep_2)))

    print('number of features expressed (in mean) above ' + str(threshold_to_keep)+' in class ' +class1_name+ ': ' + str(len(features_to_keep_1)))
    print('number of features expressed (in mean) above ' + str(threshold_to_keep)+' in class ' +class2_name+ ': ' + str(len(features_to_keep_2)))
    print('number of features in kept across both classes: ' + str(len(features_to_keep)))

    return features_to_keep

def prepare_sample_list(elements, sample_names:list = None, label_sample_dict:dict = None, sample_ordering: np.ndarray = None, sample_label_dict = None):
  '''checks elements against sample_names, a label dictionary and a sample ordering. Returns a list of sample names. At least one of
  sample_names, label_sample_dict or sample_ordering should be provided'''
  samples_to_return = []

  if sample_names is None:
    sample_names = []
  if label_sample_dict is None:
    if sample_label_dict is None:
      label_sample_dict = {}
    else:
      label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

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
        raise Warning('sample ' + element + ' is neither a valid sample, nor a label in the provided arguments - (Check for misspellings or if valid dictionaries, namings are provided)')

  return samples_to_return

def shuffle_columns(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """
    Shuffle the columns of a DataFrame randomly.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        pd.DataFrame: DataFrame with shuffled columns.
    """
    if seed is not None:
        np.random.seed(seed)
    shuffled_columns = np.random.permutation(df.columns)
    return df[shuffled_columns].copy()

def calc_corr_for_sing_feat(feature, df, feature_name = None, method = 'spearman'):

  #get name of feature and feature expressions
  if feature_name is None:
    if type(feature) == str:
      feature_name = feature
      feature = df[feature]
    else:
      feature_name = feature.name

  #choose method
  if method == 'spearman':
    corr_func = spearmanr
  else:
    raise ValueError('only supported methods are: spearman')

  #calculate correlations with features in the df
  correlation_df = pd.DataFrame(columns=df.columns)
  p_value_df = pd.DataFrame(columns=df.columns)
  for col in df.columns:
    expression_array = df[col]
    correlation, p_value = corr_func(feature, expression_array)
    correlation_df.loc[feature_name,col] = correlation
    p_value_df.loc[feature_name, col] = p_value

  return correlation_df.T, p_value_df.T

def check_labels(sample_list, sample_label_dict, sample_ordering = None):
  '''function that takes as input sample names and outputs their labels and optionally indices in a sample ordering array'''
  for i, sample in enumerate(prepare_sample_list(elements = sample_list, sample_names=sample_label_dict.index, sample_ordering=sample_ordering)):
    if sample_ordering is not None:
      print(str(np.where(sample_ordering == sample)[0]) +', '+sample +': '+sample_label_dict[sample])
    else:
      print(sample +': '+sample_label_dict[sample])

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

def divide_samples_into_classes(samples, sample_label_dict):
  '''Panasiai kaip invert_label_dict funkcija, tik cia tu dar paduodi meginiu sarasa - padeda jeigu naudoji arba ne visus meginius, arba meginiai kartojasi.'''
  new_label_sample_dict = {}
  for label in set(sample_label_dict.values()):
    new_label_sample_dict[label] = []

  for sample in samples:
    label = sample_label_dict[sample]
    new_label_sample_dict[label].append(sample)

  return new_label_sample_dict

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

def prepare_labels(samples, label_dict):
  '''creates label array for use with sklearn'''
  #label_dict should have samples as keys
  labels = np.array([label_dict[sample] for sample in samples])
  return labels

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
                                  xlim = None, ylim = None, show_legend = True, title = None, slice_for_samples:slice = None, use_numbers:bool = None, save_plot=False):
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
  plt.figure()
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
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature_1}_{feature_2}.png')
  plt.show()

def plot_two_features_use_text(df_1, feature_1, feature_2, df_2 = None, use_numbers = False, slice_for_samples:slice = None, samples_to_use:list|None = None, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, save_plot = False):
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
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature_1}_{feature_2}.png')

  plt.show()

def plot_single_feature(df, feature, samples_to_use:list|None = None, noise_level = 0.05, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None, samples_to_higlight: list|None = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, slice_for_samples:slice = None, use_numbers:bool = None, save_plot = False, random_seed = 42):
  #paima df, kuriu rows yra observazijos ir padaro vieno bruozo plot'a su trupuciu triuksmo.
  #label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values
  #slice_for_samples, use_numbers argumentai yra naudojami kitame funckijos variante - ten kur su tekstu plotinna - cia jie nereikalingi

  df = df.copy()
  #add noise column to df
  np.random.seed(random_seed)
  df['noise'] = np.random.normal(loc=0, scale=noise_level, size=len(df))

  if samples_to_use is None:
    samples_to_use = df.index.copy()
    use_all_samples = True
  else:
    use_all_samples = False

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in samples_to_use:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if use_all_samples == False:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  x_data = df[feature]
  y_data = df['noise']
                                    
  if ylim is None:
    y_min = min(y_data)
    y_max = max(y_data)
    buffer = (y_max - y_min)/10
    ylim = [y_min - buffer, y_max + buffer]

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
  plt.xlabel(feature)
  plt.ylabel('noise of level: ' + str(noise_level))
  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)
  if title is not None:
    plt.title(title)
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature}.png')
  plt.show()

def plot_single_feature_use_text(df, feature, samples_to_use:list|None = None, noise_level = 0.05, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None, samples_to_higlight: list|None = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, slice_for_samples:slice = None, use_numbers:bool = None, save_plot = False, random_seed=42):
  '''kaip kita plottinim'o funkcija, bet naudoja teksta arba skaicius vietoj tasku.'''
  #paima df, kuriu rows yra observazijos ir padaro dvieju bruozu plot'a.
  #label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values

  df = df.copy()
  #add noise column to df
  np.random.seed(random_seed)
  df['noise'] = np.random.normal(loc=0, scale=noise_level, size=len(df))

  if use_numbers and sample_ordering is None:
    raise Exception('sample_ordering should be provided if using numbers to plot')

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in df.columns:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if samples_to_use is None:
    samples_to_use = df.index.copy()
  else:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  x_data = df[feature].copy()
  y_data = df['noise'].copy()
                                    
  if ylim is None:
    y_min = min(y_data)
    y_max = max(y_data)
    buffer = (y_max - y_min)
    ylim = [y_min - buffer, y_max + buffer]

  plt.figure()
  used_labels = []
  patches_for_legend = []

  if xlim is None:
    x_min = min(x_data)
    x_max = max(x_data)
    print('feature "' +feature+ '" ranges from: ' + str(x_min) + ' to ' +str(x_max))
    buffer = (x_max - x_min)/10
    xlim = [x_min - buffer, x_max + buffer]

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
  plt.xlabel(feature)
  plt.ylabel('noise of level: ' + str(noise_level))
  if title is not None:
    plt.title(title)
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature}.png')
  plt.show()

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

def read_feature_ranking_df(url):
  #sekancios dvi funkcijos is esmes tam, kad kablelius mirnu pavadinimuose pakeistu underscorai's. Nors gal to ir nereikia?
  df = pd.read_csv(url)
  #change "_" to "," in "Feature" column:
  df['Feature'] = df['Feature'].str.replace('_', ',')
  df = df.set_index('Feature')
  return df

def write_feature_ranking_df(df, file_name):
  #change "," to "_" in "Feature" column:
  df.index = df.index.str.replace(',', '_')
  df.to_csv(file_name+'.csv')


