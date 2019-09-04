#!/usr/bin/env python3
#coding: utf-8

import re

from gutenberg import acquire
from gutenberg import cleanup

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class LineLens(text_problems.Text2ClassProblem):
  """Predict line length"""

  @property
  def approx_vocab_size(self):
    return 2**13  # ~8k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def num_classes(self):
    return 101

  def class_labels(self, data_dir):
    del data_dir
    return list(range(101))

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split


    books = [
        # bookid, skip N lines
        (19221, 223),
        (15553, 522),
    ]

    for (book_id, toskip) in books:
      text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
      lines = text.split("\n")[toskip:]
      for line in lines:
        # Any line that is all upper case is a title or author name
        if not line or line.upper() == line:
          continue

        line = re.sub("[^a-z]+", " ", line.strip().lower())
        if line:
          l = len(line)
          if l > 100:
              l = 100
          yield {
              "inputs": line,
              "label": l,
          }
