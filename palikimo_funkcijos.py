def create_labels_dict(sample_list, labels_df, slice_for_samples = slice(12), row_of_labels_df = 0):
  label_dict = {}
  #nes ten po meginio kodo dar prideda zodi "diagnosis"
  labels_df.columns = [col[:12] for col in labels_df.columns]
  for sample in sample_list:
    if sample[slice_for_samples] in labels_df.columns:
      label_dict[sample] = labels_df.loc[row_of_labels_df, sample[slice_for_samples]]
  return label_dict

def create_custom_labels(sample_groups: list, label_dict: dict, sample_ordering, reference_labels = None):
  #here samples should be inputed as indices in sample ordering. 
  #sample group is lis of lists
  sample_labels_dict = {}
  labels  = list(label_dict.keys())
  used_samples = []

  if (len(sample_groups) != len(labels)-1) and (len(sample_groups) != len(labels)):
    raise Warning('mismatch in number of groups and labels')

  for index, group in enumerate(sample_groups):
    for sample in group:
      sample_labels_dict[sample_ordering[sample]] = labels[index]
      used_samples.append(sample)
  print('index')
  if len(used_samples) < len(sample_ordering):
    unused_samples = list(set(range(len(sample_ordering))) - set(used_samples))
    for sample in unused_samples:
      sample_labels_dict[sample_ordering[sample]] = labels[index + 1]

  return sample_labels_dict

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
