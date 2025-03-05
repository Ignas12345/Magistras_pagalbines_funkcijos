import numpy as np
import pandas as pd
from Magistras_pagalbines_funkcijos.duomenu_paruosimas import *

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

  if samples_to_use is None:
    samples_to_use = df_1.index.copy()
    use_all_samples = True
  else:
    use_all_samples = False

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in samples_to_use:
      label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')
  labels = list(label_sample_dict.keys())

  if use_all_samples == False:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df_1.columns, label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(labels)

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
    samples_to_higlight = prepare_sample_list(elements = samples_to_higlight, sample_names = df_1.columns, label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)
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

  if use_numbers and sample_ordering is None:
    raise Exception('sample_ordering should be provided if using numbers to plot')

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in df_1.columns:
      label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')
  labels = list(label_sample_dict.keys())

  if samples_to_use is None:
    samples_to_use = df_1.index.copy()
  else:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df_1.columns, label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(labels)

  x_data = df_1[feature_1].copy()
  y_data = df_2[feature_2].copy()

  plt.figure()
  used_labels = []
  patches_for_legend = []

  if xlim is None:
    xlim = [min(x_data) - 1000, max(x_data) + 1000]
  if ylim is None:
    ylim = [min(y_data) - 1000, max(y_data) + 1000]

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