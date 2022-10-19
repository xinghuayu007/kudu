// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include <boost/container/flat_map.hpp>
#include <boost/container/vector.hpp>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <gtest/gtest_prod.h>

#include "kudu/cfile/cfile_reader.h"
#include "kudu/common/iterator.h"
#include "kudu/common/rowid.h"
#include "kudu/common/schema.h"
#include "kudu/gutil/gscoped_ptr.h"
#include "kudu/gutil/macros.h"
#include "kudu/gutil/map-util.h"
#include "kudu/gutil/port.h"
#include "kudu/tablet/rowset_metadata.h"
#include "kudu/util/memory/arena.h"
#include "kudu/util/status.h"

DECLARE_int32(max_encoded_key_size_bytes);

namespace boost {
template <class T>
class optional;
}  // namespace boost

namespace kudu {

class ColumnMaterializationContext;
class EncodedKey;
class KuduPartialRow;
class MemTracker;
class ScanSpec;
class SelectionVector;
struct IteratorStats;

namespace cfile {
class BloomFileReader;
}  // namespace cfile

namespace fs {
struct IOContext;
}  // namespace fs

namespace tablet {

class RowSetKeyProbe;
struct ProbeStats;

// Set of CFiles which make up the base data for a single rowset
//
// All of these files have the same number of rows, and thus the positional
// indexes can be used to seek to corresponding entries in each.
class CFileSet :
    public std::enable_shared_from_this<CFileSet>,
    public enable_make_shared<CFileSet> {
 public:
  class Iterator;

  static Status Open(std::shared_ptr<RowSetMetadata> rowset_metadata,
                     std::shared_ptr<MemTracker> bloomfile_tracker,
                     std::shared_ptr<MemTracker> cfile_reader_tracker,
                     const fs::IOContext* io_context,
                     std::shared_ptr<CFileSet>* cfile_set);

  // Create an iterator with the given projection. 'projection' must remain valid
  // for the lifetime of the returned iterator.
  std::unique_ptr<Iterator> NewIterator(const Schema* projection,
                                        const fs::IOContext* io_context) const;

  Status CountRows(const fs::IOContext* io_context, rowid_t *count) const;

  // See RowSet::GetBounds
  Status GetBounds(std::string* min_encoded_key,
                   std::string* max_encoded_key) const;

  // The on-disk size, in bytes, of this cfile set's ad hoc index.
  // Returns 0 if there is no ad hoc index.
  uint64_t AdhocIndexOnDiskSize() const;

  // The on-disk size, in bytes, of this cfile set's bloomfiles.
  // Returns 0 if there are no bloomfiles.
  uint64_t BloomFileOnDiskSize() const;

  // The size on-disk of this cfile set's data, in bytes.
  // Excludes the ad hoc index and bloomfiles.
  uint64_t OnDiskDataSize() const;

  // The size on-disk of column cfile's data, in bytes.
  uint64_t OnDiskColumnDataSize(const ColumnId& col_id) const;

  // Determine the index of the given row key.
  // Sets *idx to std::nullopt if the row is not found.
  Status FindRow(const RowSetKeyProbe& probe,
                 const fs::IOContext* io_context,
                 std::optional<rowid_t>* idx,
                 ProbeStats* stats) const;

  std::string ToString() const {
    return std::string("CFile base data in ") + rowset_metadata_->ToString();
  }

  // Check if the given row is present. If it is, sets *rowid to the
  // row's index.
  Status CheckRowPresent(const RowSetKeyProbe& probe, const fs::IOContext* io_context,
                         bool* present, rowid_t* rowid, ProbeStats* stats) const;

  // Return true if there exists a CFile for the given column ID.
  bool has_data_for_column_id(ColumnId col_id) const {
    return ContainsKey(readers_by_col_id_, col_id);
  }

  virtual ~CFileSet();

 protected:
  CFileSet(std::shared_ptr<RowSetMetadata> rowset_metadata,
           std::shared_ptr<MemTracker> bloomfile_tracker,
           std::shared_ptr<MemTracker> cfile_reader_tracker);

 private:
  friend class Iterator;

  DISALLOW_COPY_AND_ASSIGN(CFileSet);

  Status DoOpen(const fs::IOContext* io_context);
  Status OpenBloomReader(const fs::IOContext* io_context);
  Status LoadMinMaxKeys(const fs::IOContext* io_context);

  Status NewColumnIterator(ColumnId col_id,
                           cfile::CFileReader::CacheControl cache_blocks,
                           const fs::IOContext* io_context,
                           std::unique_ptr<cfile::CFileIterator>* iter) const;
  Status NewKeyIterator(const fs::IOContext* io_context,
                        std::unique_ptr<cfile::CFileIterator>* key_iter) const;

  // Return the CFileReader responsible for reading the key index.
  // (the ad-hoc reader for composite keys, otherwise the key column reader)
  cfile::CFileReader* key_index_reader() const;

  const SchemaPtr tablet_schema() const { return rowset_metadata_->tablet_schema(); }

  std::shared_ptr<RowSetMetadata> rowset_metadata_;
  std::shared_ptr<MemTracker> bloomfile_tracker_;
  std::shared_ptr<MemTracker> cfile_reader_tracker_;

  std::string min_encoded_key_;
  std::string max_encoded_key_;

  // Map of column ID to reader. These are lazily initialized as needed.
  // We use flat_map here since it's the most memory-compact while
  // still having good performance for small maps.
  typedef boost::container::flat_map<int, std::unique_ptr<cfile::CFileReader>> ReaderMap;
  ReaderMap readers_by_col_id_;

  // A file reader for an ad-hoc index, i.e. an index that sits in its own file
  // and is not embedded with the column's data blocks. This is used when the
  // index pertains to more than one column, as in the case of composite keys.
  std::unique_ptr<cfile::CFileReader> ad_hoc_idx_reader_;
  std::unique_ptr<cfile::BloomFileReader> bloom_reader_;
};


////////////////////////////////////////////////////////////

// Column-wise iterator implementation over a set of column files.
//
// This simply ties together underlying files so that they can be batched
// together, and iterated in parallel.
class CFileSet::Iterator : public ColumnwiseIterator {
 public:

  virtual Status Init(ScanSpec *spec) OVERRIDE;

  virtual Status PrepareBatch(size_t *nrows) OVERRIDE;

  virtual Status InitializeSelectionVector(SelectionVector *sel_vec) OVERRIDE;

  Status MaterializeColumn(ColumnMaterializationContext *ctx) override;

  virtual Status FinishBatch() OVERRIDE;

  virtual bool HasNext() const OVERRIDE {
    DCHECK(initted_);
    return cur_idx_ < upper_bound_idx_;
  }

  virtual std::string ToString() const OVERRIDE {
    return std::string("rowset iterator for ") + base_data_->ToString();
  }

  const Schema &schema() const OVERRIDE {
    return *projection_;
  }

  // Return the ordinal index of the next row to be returned from
  // the iterator.
  rowid_t cur_ordinal_idx() const {
    return cur_idx_;
  }

  // Collect the IO statistics for each of the underlying columns.
  virtual void GetIteratorStats(std::vector<IteratorStats> *stats) const OVERRIDE;

  virtual ~Iterator();
 private:
  DISALLOW_COPY_AND_ASSIGN(Iterator);
  FRIEND_TEST(TestCFileSet, TestRangeScan);
  friend class CFileSet;
  friend class IndexSkipScanTest;

  // 'projection' must remain valid for the lifetime of this object.
  Iterator(std::shared_ptr<CFileSet const> base_data,
           const Schema* projection,
           const fs::IOContext* io_context)
      : base_data_(std::move(base_data)),
        projection_(projection),
        initted_(false),
        cur_idx_(0),
        prepared_count_(0),
        io_context_(io_context),
        arena_(FLAGS_max_encoded_key_size_bytes) {
        }

  // Fill in col_iters_ for each of the requested columns.
  Status CreateColumnIterators(const ScanSpec* spec);

  Status OptimizePKPredicates(ScanSpec* spec);

  // Look for a predicate which can be converted into a range scan using the key
  // column's index. If such a predicate exists, remove it from the scan spec and
  // store it in member fields.
  Status PushdownRangeScanPredicate(ScanSpec *spec);

  void Unprepare();

  //
  // Index Skip Scan overview
  //
  //   In general, for a scan predicated on a subset of columns of a composite
  //   (a.k.a. multi-column) primary key, it's not possible to use the B-tree
  //   index to simply seek to the appropriate range of rows and then scan only
  //   the necessary ones. The only exception is when a scan is predicated
  //   on the first K columns of a composite N-column primary key, where K < N,
  //   K >= 1. For all other cases, a simplistic approach mandates a full tablet
  //   scan, materializing every row to evaluate the predicates on K columns.
  //
  //   However, it's possible to devise a synthetic approach that would avoid
  //   scanning of all tablet's rows. The idea is to use the B-tree index to
  //   seek to the start of distinctly prefixed row ranges, and for each
  //   distinctly prefixed range perform scanning only within the sub-range
  //   defined by the first predicate. So, the rest of predicates on the (K - 1)
  //   columns are evaluated only for the latter sub-range.
  //
  //   This approach is dubbed "Index Skip Scan".
  //
  // Definitions
  //
  //   "predicate column"
  //     Leftmost, non-leading primary key column with a predicate.
  //
  //   "predicate value"
  //     Value that the "predicate column" is predicated on. Since only equality
  //     predicates are supported, there is only a single value of interest.
  //
  //   "prefix column"
  //     The collective columns to the left of the "predicate column".
  //
  //   "prefix key"
  //     Any value in the "prefix column".
  //
  // Details
  //
  //   In order to keep track of the number of rows that satisfy the
  //   "predicate value" wrt a distinct prefix key we first seek to the first
  //   such row in the index column and this corresponding value is called
  //   "lower bound" key. Next, we seek to the row that is right after the last
  //   row that satisfies the "predicate value" wrt to the distinct prefix key,
  //   and this corresponding value is called "upper bound" key.
  //


  // By the definition (see above), the skip scan optimization is possible
  // if there exists an equality predicate on any subset of the composite
  // primary key's columns not including the very first one.
  void TryEnableSkipScan(const ScanSpec& spec);

  // Decode the currently-seeked key into 'enc_key'.
  Status DecodeCurrentKey(gscoped_ptr<EncodedKey>* enc_key);

  // This function is used to place the validx_iter_ at the next greater prefix key.
  Status SeekToNextPrefixKey(size_t num_prefix_cols, bool cache_seeked_value);

  // Seek to the next row that matches the predicate and has the same prefix
  // as that of `enc_key`.
  Status SeekToRowWithCurPrefixMatchingPred(const gscoped_ptr<EncodedKey>& enc_key);

  // Build the key with the same prefix as 'cur_enc_key', that has
  // 'skip_scan_predicate_value_' in its predicate column,
  // and the minimum possible value for all other columns.
  Status BuildKeyWithPredicateVal(const gscoped_ptr<EncodedKey>& cur_enc_key,
                                  KuduPartialRow* p_row,
                                  gscoped_ptr<EncodedKey>* enc_key);

  // Returns true if the given encoded key matches the skip scan predicate.
  bool CheckPredicateMatch(const gscoped_ptr<EncodedKey>& enc_key) const;

  // Check if the column values in the range corresponding to the given
  // inclusive column id range [start_col_id, end_col_id] are equal between the
  // two given keys.
  bool KeyColumnsMatch(const gscoped_ptr<EncodedKey>& key1,
                       const gscoped_ptr<EncodedKey>& key2,
                       int start_col_id, int end_col_id) const;

  // This method implements a "skip-scan" optimization, allowing a scan to use
  // the primary key index to efficiently seek to matching rows where there are
  // predicates on compound key columns that do not necessarily include the
  // leading column of the primary key. At the time of writing, only a single
  // equality predicate is supported, although the algorithm can support ranges
  // of values.
  //
  // This method should be invoked during the PrepareBatch() phase of the row
  // iterator lifecycle.
  //
  // This method assumes exclusive access to key_iter.
  //
  // The in-out parameter 'remaining' refers to the number of rows remaining to
  // scan. When this method is invoked, 'remaining' should contain the maximum
  // number of remaining rows available to scan. Once this method returns,
  // 'remaining' will contain the number of rows to scan to consume the
  // available matching rows according to the equality predicate. Note:
  // 'remaining' will always be at least 1, although it is a TODO to allow it
  // to be 0 (0 violates CHECK conditions elsewhere in the scan code).
  //
  // Currently, skip scan will be dynamically disabled when the number of seeks
  // for distinct prefix keys exceeds sqrt(#total rows). We use sqrt(#total rows)
  // as a cutoff because based on performance tests on upto 10M rows per tablet,
  // the scan time for skip scan is the same as that of the current flow until
  // #seeks = sqrt(#total_rows). Further increase in #seeks leads to a drop in
  // skip scan performance wrt the current flow. This cutoff value is stored in
  // 'skip_scan_num_seeks_cutoff_'.
  //
  // Preconditions upon entering this method:
  //   * key_iter_ is not NULL
  //
  // Postconditions upon exiting this method:
  //   * cur_idx_ is updated to the row_id of the next row(containing the next
  //     higher distinct prefix) to scan
  //   * 'remaining' stores the number the entries to be scanned in the current
  //     scan range.
  //
  // See the .cc file for details on the approach and the implementation.
  Status SkipToNextScan(size_t* remaining);

  // Prepare the given column if not already prepared.
  Status PrepareColumn(ColumnMaterializationContext* ctx);

  const std::shared_ptr<CFileSet const> base_data_;
  const Schema* projection_;

  // Iterator for the key column in the underlying data.
  std::unique_ptr<cfile::CFileIterator> key_iter_;
  std::vector<std::unique_ptr<cfile::ColumnIterator>> col_iters_;

  bool initted_;

  size_t cur_idx_;
  size_t prepared_count_;

  // The total number of rows in the file
  rowid_t row_count_;

  // Lower bound (inclusive) and upper bound (exclusive) for this iterator, in terms of
  // ordinal row indexes.
  // Both of these bounds are always set (even if there is no predicate).
  // If there is no predicate, then the bounds will be [0, row_count_]
  rowid_t lower_bound_idx_;
  rowid_t upper_bound_idx_;

  const fs::IOContext* io_context_;

  // The underlying columns are prepared lazily, so that if a column is never
  // materialized, it doesn't need to be read off disk.
  //
  // The list of columns which have been prepared (and thus must be 'finished'
  // at the end of the current batch). These are pointers into the same iterators
  // stored in 'col_iters_'.
  std::vector<cfile::ColumnIterator*> prepared_iters_;

  // Flag for whether index skip scan is used.
  bool use_skip_scan_ = false;

  // Store equality predicate value to use skip scan.
  const void* skip_scan_predicate_value_;

  // Column id of the "predicate_column".
  int skip_scan_predicate_column_id_;

  // Row id of the next row that does not match the predicate.
  // This is an exclusive upper bound on our scan range.
  // A value of -1 indicates that the upper bound is not known.
  int64_t skip_scan_upper_bound_idx_ = -1;

  // Store the number of seeks for distinct prefixes in
  // index skip scan.
  int64_t skip_scan_num_seeks_ = 0;

  // Store the cutoff on the number of skip scan seeks before disabling skip scan.
  int64_t skip_scan_num_seeks_cutoff_;

  // Whether the skip scan optimization has searched the current prefix for a predicate match
  // or whether the prefix has changed since its last check.
  bool skip_scan_searched_cur_prefix_ = true;

  // Buffer to store the pointer to encoded key during skip scan.
  Arena arena_;
};

} // namespace tablet
} // namespace kudu
