def create_labels_dict(sample_df, labels_df, row_of_labels_df = 0):
  label_dict = {}
  labels_df.columns = [col[:12] for col in labels_df.columns]
  for col in sample_df.columns:
    if col[:12] in labels_df.columns:
      label_dict[col[:16]] = labels_df.loc[row_of_labels_df, col[:12]]
  return label_dict