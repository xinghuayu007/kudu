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
#include "kudu/tablet/cfile_set.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/container/flat_map.hpp>
#include <boost/container/vector.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "kudu/cfile/bloomfile.h"
#include "kudu/cfile/cfile_reader.h"
#include "kudu/cfile/cfile_util.h"
#include "kudu/common/column_materialization_context.h"
#include "kudu/common/column_predicate.h"
#include "kudu/common/columnblock.h"
#include "kudu/common/encoded_key.h"
#include "kudu/common/iterator_stats.h"
#include "kudu/common/partial_row.h"
#include "kudu/common/row.h"
#include "kudu/common/rowblock.h"
#include "kudu/common/rowid.h"
#include "kudu/common/scan_spec.h"
#include "kudu/common/schema.h"
#include "kudu/common/types.h"
#include "kudu/fs/block_id.h"
#include "kudu/fs/block_manager.h"
#include "kudu/fs/fs_manager.h"
#include "kudu/gutil/dynamic_annotations.h"
#include "kudu/gutil/macros.h"
#include "kudu/gutil/map-util.h"
#include "kudu/gutil/port.h"
#include "kudu/gutil/stringprintf.h"
#include "kudu/gutil/strings/stringpiece.h"
#include "kudu/gutil/strings/substitute.h"
#include "kudu/tablet/diskrowset.h"
#include "kudu/tablet/rowset.h"
#include "kudu/tablet/rowset_metadata.h"
#include "kudu/util/flag_tags.h"
#include "kudu/util/logging.h"
#include "kudu/util/memory/arena.h"
#include "kudu/util/slice.h"
#include "kudu/util/status.h"

DEFINE_bool(consult_bloom_filters, true, "Whether to consult bloom filters on row presence checks");
TAG_FLAG(consult_bloom_filters, hidden);

DECLARE_bool(rowset_metadata_store_keys);

DEFINE_bool(enable_skip_scan, false, "Whether to enable index skip scan");
TAG_FLAG(enable_skip_scan, experimental);

DEFINE_int32(skip_scan_short_circuit_loops, 100,
             "Max number of skip attempts the skip scan optimization should make before "
             "returning control to the main loop");
TAG_FLAG(skip_scan_short_circuit_loops, hidden);
TAG_FLAG(skip_scan_short_circuit_loops, advanced);

using kudu::cfile::BloomFileReader;
using kudu::cfile::CFileIterator;
using kudu::cfile::CFileReader;
using kudu::cfile::ColumnIterator;
using kudu::cfile::ReaderOptions;
using kudu::cfile::DefaultColumnValueIterator;
using kudu::fs::IOContext;
using kudu::fs::ReadableBlock;
using std::optional;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using strings::Substitute;

namespace kudu {

class MemTracker;

namespace tablet {

////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////

static Status OpenReader(FsManager* fs,
                         shared_ptr<MemTracker> cfile_reader_tracker,
                         const BlockId& block_id,
                         const IOContext* io_context,
                         unique_ptr<CFileReader>* new_reader) {
  unique_ptr<ReadableBlock> block;
  RETURN_NOT_OK(fs->OpenBlock(block_id, &block));

  ReaderOptions opts;
  opts.parent_mem_tracker = std::move(cfile_reader_tracker);
  opts.io_context = io_context;
  return CFileReader::OpenNoInit(std::move(block),
                                 std::move(opts),
                                 new_reader);
}

////////////////////////////////////////////////////////////
// CFile Base
////////////////////////////////////////////////////////////

CFileSet::CFileSet(shared_ptr<RowSetMetadata> rowset_metadata,
                   shared_ptr<MemTracker> bloomfile_tracker,
                   shared_ptr<MemTracker> cfile_reader_tracker)
    : rowset_metadata_(std::move(rowset_metadata)),
      bloomfile_tracker_(std::move(bloomfile_tracker)),
      cfile_reader_tracker_(std::move(cfile_reader_tracker)) {
}

CFileSet::~CFileSet() {
}

Status CFileSet::Open(shared_ptr<RowSetMetadata> rowset_metadata,
                      shared_ptr<MemTracker> bloomfile_tracker,
                      shared_ptr<MemTracker> cfile_reader_tracker,
                      const IOContext* io_context,
                      shared_ptr<CFileSet>* cfile_set) {
  auto cfs(CFileSet::make_shared(
      std::move(rowset_metadata),
      std::move(bloomfile_tracker),
      std::move(cfile_reader_tracker)));
  RETURN_NOT_OK(cfs->DoOpen(io_context));

  cfile_set->swap(cfs);
  return Status::OK();
}

Status CFileSet::DoOpen(const IOContext* io_context) {
  RETURN_NOT_OK(OpenBloomReader(io_context));

  // Lazily open the column data cfiles. Each one will be fully opened
  // later, when the first iterator seeks for the first time.
  RowSetMetadata::ColumnIdToBlockIdMap block_map = rowset_metadata_->GetColumnBlocksById();
  for (const RowSetMetadata::ColumnIdToBlockIdMap::value_type& e : block_map) {
    ColumnId col_id = e.first;
    DCHECK(!ContainsKey(readers_by_col_id_, col_id)) << "already open";

    unique_ptr<CFileReader> reader;
    RETURN_NOT_OK(OpenReader(rowset_metadata_->fs_manager(),
                             cfile_reader_tracker_,
                             rowset_metadata_->column_data_block_for_col_id(col_id),
                             io_context,
                             &reader));
    readers_by_col_id_[col_id] = std::move(reader);
    VLOG(1) << "Successfully opened cfile for column id " << col_id
            << " in " << rowset_metadata_->ToString();
  }
  readers_by_col_id_.shrink_to_fit();

  if (rowset_metadata_->has_adhoc_index_block()) {
    RETURN_NOT_OK(OpenReader(rowset_metadata_->fs_manager(),
                             cfile_reader_tracker_,
                             rowset_metadata_->adhoc_index_block(),
                             io_context,
                             &ad_hoc_idx_reader_));
  }

  // If the user specified to store the min/max keys in the rowset metadata,
  // fetch them. Otherwise, load the min and max keys from the key reader.
  if (FLAGS_rowset_metadata_store_keys && rowset_metadata_->has_encoded_keys()) {
    min_encoded_key_ = rowset_metadata_->min_encoded_key();
    max_encoded_key_ = rowset_metadata_->max_encoded_key();
  } else {
    RETURN_NOT_OK(LoadMinMaxKeys(io_context));
  }
  // Verify the loaded keys are valid.
  if (Slice(min_encoded_key_) > max_encoded_key_) {
    return Status::Corruption(Substitute("Min key $0 > max key $1",
                                         KUDU_REDACT(Slice(min_encoded_key_).ToDebugString()),
                                         KUDU_REDACT(Slice(max_encoded_key_).ToDebugString())),
                              ToString());
  }

  return Status::OK();
}

Status CFileSet::OpenBloomReader(const IOContext* io_context) {
  FsManager* fs = rowset_metadata_->fs_manager();
  unique_ptr<ReadableBlock> block;
  RETURN_NOT_OK(fs->OpenBlock(rowset_metadata_->bloom_block(), &block));

  ReaderOptions opts;
  opts.io_context = io_context;
  opts.parent_mem_tracker = bloomfile_tracker_;
  Status s = BloomFileReader::OpenNoInit(std::move(block),
                                         std::move(opts),
                                         &bloom_reader_);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to open bloom file in " << rowset_metadata_->ToString() << ": "
                 << s.ToString();
    // Continue without bloom.
  }

  return Status::OK();
}

Status CFileSet::LoadMinMaxKeys(const IOContext* io_context) {
  CFileReader* key_reader = key_index_reader();
  RETURN_NOT_OK(key_index_reader()->Init(io_context));
  if (!key_reader->GetMetadataEntry(DiskRowSet::kMinKeyMetaEntryName, &min_encoded_key_)) {
    return Status::Corruption("No min key found", ToString());
  }
  if (!key_reader->GetMetadataEntry(DiskRowSet::kMaxKeyMetaEntryName, &max_encoded_key_)) {
    return Status::Corruption("No max key found", ToString());
  }
  return Status::OK();
}

CFileReader* CFileSet::key_index_reader() const {
  if (ad_hoc_idx_reader_) {
    return ad_hoc_idx_reader_.get();
  }
  // If there is no special index cfile, then we have a non-compound key
  // and we can just use the key column.
  // This is always the first column listed in the tablet schema.
  int key_col_id = tablet_schema()->column_id(0);
  return FindOrDie(readers_by_col_id_, key_col_id).get();
}

Status CFileSet::NewColumnIterator(ColumnId col_id,
                                   CFileReader::CacheControl cache_blocks,
                                   const fs::IOContext* io_context,
                                   unique_ptr<CFileIterator>* iter) const {
  return FindOrDie(readers_by_col_id_, col_id)->NewIterator(iter, cache_blocks,
                                                            io_context);
}

unique_ptr<CFileSet::Iterator> CFileSet::NewIterator(
    const Schema* projection,
    const IOContext* io_context) const {
  return unique_ptr<CFileSet::Iterator>(
      new CFileSet::Iterator(shared_from_this(), projection, io_context));
}

Status CFileSet::CountRows(const IOContext* io_context, rowid_t *count) const {
  RETURN_NOT_OK(key_index_reader()->Init(io_context));
  return key_index_reader()->CountRows(count);
}

Status CFileSet::GetBounds(string* min_encoded_key,
                           string* max_encoded_key) const {
  *min_encoded_key = min_encoded_key_;
  *max_encoded_key = max_encoded_key_;
  return Status::OK();
}

uint64_t CFileSet::AdhocIndexOnDiskSize() const {
  if (ad_hoc_idx_reader_) {
    return ad_hoc_idx_reader_->file_size();
  }
  return 0;
}

uint64_t CFileSet::BloomFileOnDiskSize() const {
  return bloom_reader_->FileSize();
}

uint64_t CFileSet::OnDiskDataSize() const {
  uint64_t ret = 0;
  for (const auto& e : readers_by_col_id_) {
    ret += e.second->file_size();
  }
  return ret;
}

uint64_t CFileSet::OnDiskColumnDataSize(const ColumnId& col_id) const {
  return FindOrDie(readers_by_col_id_, col_id)->file_size();
}

Status CFileSet::FindRow(const RowSetKeyProbe &probe,
                         const IOContext* io_context,
                         optional<rowid_t>* idx,
                         ProbeStats* stats) const {
  if (FLAGS_consult_bloom_filters) {
    // Fully open the BloomFileReader if it was lazily opened earlier.
    //
    // If it's already initialized, this is a no-op.
    RETURN_NOT_OK(bloom_reader_->Init(io_context));

    stats->blooms_consulted++;
    bool present;
    Status s = bloom_reader_->CheckKeyPresent(probe.bloom_probe(), io_context, &present);
    if (s.ok() && !present) {
      idx->reset();
      return Status::OK();
    }
    if (!s.ok()) {
      KLOG_EVERY_N_SECS(WARNING, 1) << Substitute("Unable to query bloom in $0: $1",
          rowset_metadata_->bloom_block().ToString(), s.ToString());
      if (PREDICT_FALSE(s.IsDiskFailure())) {
        // If the bloom lookup failed because of a disk failure, return early
        // since I/O to the tablet should be stopped.
        return s;
      }
      // Continue with the slow path
    }
  }

  stats->keys_consulted++;
  unique_ptr<CFileIterator> key_iter;
  RETURN_NOT_OK(NewKeyIterator(io_context, &key_iter));

  bool exact;
  Status s = key_iter->SeekAtOrAfter(probe.encoded_key(),
      /* cache_seeked_value= */ false,
      /* exact_match= */ &exact);
  if (s.IsNotFound() || (s.ok() && !exact)) {
    idx->reset();
    return Status::OK();
  }
  RETURN_NOT_OK(s);

  *idx = key_iter->GetCurrentOrdinal();
  return Status::OK();
}

Status CFileSet::CheckRowPresent(const RowSetKeyProbe& probe, const IOContext* io_context,
                                 bool* present, rowid_t* rowid, ProbeStats* stats) const {
  optional<rowid_t> opt_rowid;
  RETURN_NOT_OK(FindRow(probe, io_context, &opt_rowid, stats));
  *present = opt_rowid.has_value();
  if (*present) {
  // Suppress false positive about 'opt_rowid' used when uninitialized.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    *rowid = *opt_rowid;
#pragma GCC diagnostic pop
  }
  return Status::OK();
}

Status CFileSet::NewKeyIterator(const IOContext* io_context,
                                unique_ptr<CFileIterator>* key_iter) const {
  RETURN_NOT_OK(key_index_reader()->Init(io_context));
  return key_index_reader()->NewIterator(key_iter, CFileReader::CACHE_BLOCK, io_context);
}

////////////////////////////////////////////////////////////
// Iterator
////////////////////////////////////////////////////////////
CFileSet::Iterator::~Iterator() {
}

Status CFileSet::Iterator::CreateColumnIterators(const ScanSpec* spec) {
  DCHECK_EQ(0, col_iters_.size());
  vector<unique_ptr<ColumnIterator>> ret_iters;
  ret_iters.reserve(projection_->num_columns());

  CFileReader::CacheControl cache_blocks = CFileReader::CACHE_BLOCK;
  if (spec && !spec->cache_blocks()) {
    cache_blocks = CFileReader::DONT_CACHE_BLOCK;
  }

  for (int proj_col_idx = 0;
       proj_col_idx < projection_->num_columns();
       proj_col_idx++) {
    ColumnId col_id = projection_->column_id(proj_col_idx);

    if (!base_data_->has_data_for_column_id(col_id)) {
      // If we have no data for a column, most likely it was added via an ALTER
      // operation after this CFileSet was flushed. In that case, we're guaranteed
      // that it is either NULLable, or has a "read-default". Otherwise, consider it a corruption.
      const ColumnSchema& col_schema = projection_->column(proj_col_idx);
      if (PREDICT_FALSE(!col_schema.is_nullable() && !col_schema.has_read_default())) {
        return Status::Corruption(Substitute("column $0 has no data in rowset $1",
                                             col_schema.ToString(), base_data_->ToString()));
      }
      ret_iters.emplace_back(new DefaultColumnValueIterator(col_schema.type_info(),
                                                            col_schema.read_default_value()));
      continue;
    }
    unique_ptr<CFileIterator> iter;
    RETURN_NOT_OK_PREPEND(base_data_->NewColumnIterator(col_id, cache_blocks, io_context_, &iter),
                          Substitute("could not create iterator for column $0",
                                     projection_->column(proj_col_idx).ToString()));
    ret_iters.emplace_back(std::move(iter));
  }

  col_iters_.swap(ret_iters);
  prepared_iters_.reserve(col_iters_.size());
  return Status::OK();
}

Status CFileSet::Iterator::Init(ScanSpec *spec) {
  CHECK(!initted_);

  RETURN_NOT_OK(base_data_->CountRows(io_context_, &row_count_));
  CHECK_GT(row_count_, 0);

  // Setup key iterator.
  RETURN_NOT_OK(base_data_->NewKeyIterator(io_context_, &key_iter_));

  // Setup column iterators.
  RETURN_NOT_OK(CreateColumnIterators(spec));

  lower_bound_idx_ = 0;
  upper_bound_idx_ = row_count_;
  // RETURN_NOT_OK(OptimizePKPredicates(spec));
  if (spec != nullptr && spec->CanShortCircuit()) {
    lower_bound_idx_ = row_count_;
    spec->RemovePredicates();
  } else {
    // If there is a range predicate on the key column, push that down into an
    // ordinal range.
    RETURN_NOT_OK(PushdownRangeScanPredicate(spec));
  }
  if (!spec->predicates().empty()) {
    TryEnableSkipScan(*spec);
  }
  initted_ = true;

  // Don't actually seek -- we'll seek when we first actually read the
  // data.
  cur_idx_ = lower_bound_idx_;
  Unprepare(); // Reset state.
  return Status::OK();
}

Status CFileSet::Iterator::OptimizePKPredicates(ScanSpec* spec) {
  if (spec == nullptr) {
    // No predicate.
    return Status::OK();
  }

  const EncodedKey* lb_key = spec->lower_bound_key();
  const EncodedKey* ub_key = spec->exclusive_upper_bound_key();
  EncodedKey* implicit_lb_key = nullptr;
  EncodedKey* implicit_ub_key = nullptr;
  bool modify_lower_bound_key = false;
  bool modify_upper_bound_key = false;
  const Schema& tablet_schema = *base_data_->tablet_schema();

  if (!lb_key || lb_key->encoded_key() < base_data_->min_encoded_key_) {
    RETURN_NOT_OK(EncodedKey::DecodeEncodedString(
        tablet_schema, &arena_, base_data_->min_encoded_key_, &implicit_lb_key));
    spec->SetLowerBoundKey(implicit_lb_key);
    modify_lower_bound_key = true;
  }

  RETURN_NOT_OK(EncodedKey::DecodeEncodedString(
      tablet_schema, &arena_, base_data_->max_encoded_key_, &implicit_ub_key));
  Status s = EncodedKey::IncrementEncodedKey(tablet_schema, &implicit_ub_key, &arena_);
  // Reset the exclusive_upper_bound_key only when we can get a valid and smaller upper bound key.
  // In the case IncrementEncodedKey return ERROR status due to allocation fails or no
  // lexicographically greater key exists, we fall back to scan the rowset without optimizing the
  // upper bound PK, we may scan more rows but we will still get the right result.
  if (s.ok() && (!ub_key || ub_key->encoded_key() > implicit_ub_key->encoded_key())) {
    spec->SetExclusiveUpperBoundKey(implicit_ub_key);
    modify_upper_bound_key = true;
  }

  if (modify_lower_bound_key || modify_upper_bound_key) {
    spec->UnifyPrimaryKeyBoundsAndColumnPredicates(tablet_schema, &arena_, true);
  }
  return Status::OK();
}

Status CFileSet::Iterator::PushdownRangeScanPredicate(ScanSpec* spec) {
  if (spec == nullptr) {
    // No predicate.
    return Status::OK();
  }

  Schema key_schema_for_vlog;
  if (VLOG_IS_ON(1)) {
    key_schema_for_vlog = base_data_->tablet_schema()->CreateKeyProjection();
  }

  const auto* lb_key = spec->lower_bound_key();
  if (lb_key && lb_key->encoded_key() > base_data_->min_encoded_key_) {
    bool exact;
    Status s = key_iter_->SeekAtOrAfter(*spec->lower_bound_key(),
        /* cache_seeked_value= */ false,
        /* exact_match= */ &exact);
    if (s.IsNotFound()) {
      // The lower bound is after the end of the key range.
      // Thus, no rows will pass the predicate, so we set the lower bound
      // to the end of the file.
      lower_bound_idx_ = row_count_;
      return Status::OK();
    }
    RETURN_NOT_OK(s);

    lower_bound_idx_ = std::max(lower_bound_idx_, key_iter_->GetCurrentOrdinal());
    VLOG(1) << "Pushed lower bound value "
            << lb_key->Stringify(key_schema_for_vlog)
            << " as row_idx >= " << lower_bound_idx_;
  }
  const auto* ub_key = spec->exclusive_upper_bound_key();
  if (ub_key && ub_key->encoded_key() <= base_data_->max_encoded_key_) {
    bool exact;
    Status s = key_iter_->SeekAtOrAfter(*spec->exclusive_upper_bound_key(),
        /* cache_seeked_value= */ false,
        /* exact_match= */ &exact);
    if (PREDICT_FALSE(s.IsNotFound())) {
      LOG(DFATAL) << "CFileSet indicated upper bound was within range, but "
                  << "key iterator could not seek. "
                  << "CFileSet upper_bound = "
                  << KUDU_REDACT(Slice(base_data_->max_encoded_key_).ToDebugString())
                  << ", enc_key = "
                  << KUDU_REDACT(ub_key->encoded_key().ToDebugString());
    } else {
      RETURN_NOT_OK(s);

      rowid_t cur = key_iter_->GetCurrentOrdinal();
      upper_bound_idx_ = std::min(upper_bound_idx_, cur);

      VLOG(1) << "Pushed upper bound value "
              << ub_key->Stringify(key_schema_for_vlog)
              << " as row_idx < " << upper_bound_idx_;
    }
  }
  return Status::OK();
}

void CFileSet::Iterator::Unprepare() {
  prepared_count_ = 0;
  prepared_iters_.clear();
}

void CFileSet::Iterator::TryEnableSkipScan(const ScanSpec& spec) {
  const SchemaPtr& schema = base_data_->tablet_schema();
  const auto num_key_cols = schema->num_key_columns();

  if (!FLAGS_enable_skip_scan || num_key_cols <= 1) {
    use_skip_scan_ = false;
    return;
  }

  // Do not enable skip scan if primary key push down has already occurred.
  if (lower_bound_idx_ != 0 || upper_bound_idx_ != row_count_) {
    std::cout << "wangixu-lower_bound_idx_:" << lower_bound_idx_ << " upper_bound_idx_:" << upper_bound_idx_ << std::endl;
    use_skip_scan_ = false;
    return;
  }

  bool non_prefix_key_column_pred_exists = false;

  // Tracks the minimum column id for the non-prefix column predicate(s).
  // Initialize the min column id to an upperbound value.
  int min_non_prefix_col_id = num_key_cols;

  // Tracks the equality predicate value for "min_non_prefix_col_id"
  const void* min_non_prefix_pred_value;

  for (const auto& col_and_pred : spec.predicates()) {
    const string& col_name = col_and_pred.first;
    const ColumnPredicate& pred = col_and_pred.second;
    // Get the column id from the predicate
    StringPiece sp(reinterpret_cast<const char*>(col_name.data()), col_name.size());
    int col_id = schema->find_column(sp);

    if (col_id == 0 or col_id == -1) {
      // col_id = 0 implies that predicate exists on the first PK column.
      // col_id = -1 implies that column is not found in the schema.
      if (col_id == -1) {
        LOG(WARNING) << col_name << " column is not found in schema";
      }
      use_skip_scan_ = false;
      return;
    }

    if (schema->is_key_column(col_id) &&
        pred.predicate_type() == PredicateType::Equality) {
      non_prefix_key_column_pred_exists = true;
      if (col_id < min_non_prefix_col_id) {
        min_non_prefix_col_id = col_id;
        min_non_prefix_pred_value = pred.raw_lower();
      }
    }
  }

  if (non_prefix_key_column_pred_exists) {
    use_skip_scan_ = true;

    // Store the predicate column id.
    skip_scan_predicate_column_id_ = min_non_prefix_col_id;

    // Store the predicate value.
    skip_scan_predicate_value_ = min_non_prefix_pred_value;

    // Store the cutoff on the number of skip scan seeks.
    skip_scan_num_seeks_cutoff_ = static_cast<int64_t>(sqrt(row_count_));
  }
}

Status CFileSet::Iterator::DecodeCurrentKey(gscoped_ptr<EncodedKey>* enc_key) {
  EncodedKey* enc_key_tmp = nullptr;
  RETURN_NOT_OK_PREPEND(EncodedKey::DecodeEncodedString(
      *(base_data_->tablet_schema().get()), &arena_,
      key_iter_->GetCurrentValue(), &enc_key_tmp),
      "Failed to decode current value from primary key index");
  enc_key->reset(enc_key_tmp);
  return Status::OK();
}

// If 'cache_seeked_value' is true:
// 1. The validx_iter_ will store the value seeked to.
// 2. In this case, prior to calling key_iter->SeekAtOrAfter(), the search key is
//    modified such that the predicate column is set to the predicate value and the
//    succeeding columns are set to their minimum possible values. This is done to
//    make sure that after calling key_iter->SeekAtOrAfter() the key_iter is correctly
//    placed at the first occurrence of the row (if it exists) that matches the next
//    prefix key with the predicate value.
Status CFileSet::Iterator::SeekToNextPrefixKey(size_t num_prefix_cols, bool cache_seeked_value) {
  gscoped_ptr<EncodedKey> enc_key;
  RETURN_NOT_OK(DecodeCurrentKey(&enc_key));
  EncodedKey* enc_key_tmp = enc_key.get();
  // Increment the prefix key which we consider to be the first "num_prefix_cols"
  // columns of the cached value obtained after the previous seek in the primary key index.
  RETURN_NOT_OK(EncodedKey::IncrementEncodedKeyColumns(
      *(base_data_->tablet_schema().get()),
      num_prefix_cols, &arena_, &enc_key_tmp));
  enc_key.reset(enc_key_tmp);
  if (cache_seeked_value) {
    // Set the predicate column to the predicate value in case we can find a
    // predicate match in one search. As a side effect, BuildKeyWithPredicateVal()
    // sets minimum values on the columns after the predicate value, which is
    // required for correctness here.
    KuduPartialRow partial_row(base_data_->tablet_schema().get());
    RETURN_NOT_OK(BuildKeyWithPredicateVal(enc_key, &partial_row, &enc_key));
  }
  return key_iter_->SeekAtOrAfter(*enc_key,
      /* cache_seeked_value= */ cache_seeked_value,
      /* exact_match= */ nullptr);
}

Status CFileSet::Iterator::SeekToRowWithCurPrefixMatchingPred(
    const gscoped_ptr<EncodedKey>& enc_key) {
  // Check to see if the current key matches the predicate value. If so, then
  // there is no need to seek forward.
  if (CheckPredicateMatch(enc_key)) {
    return Status::OK();
  }

  // If we got this far, the current key doesn't match the predicate, so search
  // for the next key that matches the current prefix and predicate.
  KuduPartialRow partial_row(base_data_->tablet_schema().get());
  gscoped_ptr<EncodedKey> key_with_pred_value;
  RETURN_NOT_OK(BuildKeyWithPredicateVal(enc_key, &partial_row, &key_with_pred_value));
  return key_iter_->SeekAtOrAfter(*key_with_pred_value,
      /* cache_seeked_value= */ true,
      /* exact_match= */ nullptr);
}

// TODO(anupama): to support in-range predicates, generalize this to build a key with
// the same prefix as cur_enc_key, a predicate column populated with the lower bound of
// the predicate values, and the minimum value for all other columns.
Status CFileSet::Iterator::BuildKeyWithPredicateVal(
    const gscoped_ptr<EncodedKey> &cur_enc_key, KuduPartialRow *p_row,
    gscoped_ptr<EncodedKey> *enc_key) {

  int col_id = 0;
  // Build a new partial row with the current prefix key value and the
  // predicate value.
  for (auto const& value : cur_enc_key->raw_keys()) {
    if (col_id < skip_scan_predicate_column_id_) {
      const uint8_t *data = reinterpret_cast<const uint8_t *>(value);
      RETURN_NOT_OK(p_row->Set(col_id, data));
    } else {
      // Set the predicate value.
      const uint8_t *suffix_col_value =
          reinterpret_cast<const uint8_t *>(skip_scan_predicate_value_);
      RETURN_NOT_OK(p_row->Set(skip_scan_predicate_column_id_, suffix_col_value));
      break;
    }
    col_id++;
  }

  // Fill the values after the predicate column id with their
  // minimum possible values.
  ContiguousRow cont_row(base_data_->tablet_schema().get(), p_row->row_data_);
  for (size_t i = skip_scan_predicate_column_id_ + 1;
       i < base_data_->tablet_schema()->num_key_columns(); i++) {
    const ColumnSchema& col = cont_row.schema()->column(i);
    col.type_info()->CopyMinValue(cont_row.mutable_cell_ptr(i));
  }
  // Build the new encoded key.
  ConstContiguousRow const_row(cont_row);
  gscoped_ptr<EncodedKey> new_enc_key(EncodedKey::FromContiguousRow(const_row, &arena_));
  *enc_key = new_enc_key.Pass();
  return Status::OK();
}

bool CFileSet::Iterator::CheckPredicateMatch(const gscoped_ptr<EncodedKey>& enc_key) const {
  return base_data_->tablet_schema()->column(skip_scan_predicate_column_id_).Compare(
      skip_scan_predicate_value_,
      enc_key->raw_keys()[skip_scan_predicate_column_id_]) == 0;
}

bool CFileSet::Iterator::KeyColumnsMatch(const gscoped_ptr<EncodedKey>& key1,
                                         const gscoped_ptr<EncodedKey>& key2,
                                         int start_col_id, int end_col_id) const {
  const auto& schema = base_data_->tablet_schema();
  for (int col_id = start_col_id; col_id <= end_col_id; col_id++) {
    if (schema->column(col_id).Compare(key1->raw_keys()[col_id], key2->raw_keys()[col_id]) != 0) {
      return false;
    }
  }
  return true;
}

Status CFileSet::Iterator::SkipToNextScan(size_t *remaining) {
  // Keep scanning if we're still in the range that needs scanning from our
  // previous seek. static_cast required because cur_idx_ is unsigned and the
  // upper bound index can be negative.
  if (static_cast<int64_t>(cur_idx_) < skip_scan_upper_bound_idx_) {
    *remaining = std::max<int64_t>(skip_scan_upper_bound_idx_ - cur_idx_, 1);
    return Status::OK();
  }

  // This is a three seek approach for index skip scan implementation:
  // 1. Search within the ad-hoc index for the next distinct prefix
  //    (set of keys prior to the predicate column).
  //    Searching is done using validx_iter_.
  // 2. Read that distinct prefix from the ad-hoc index, append the predicate value
  //    and minimum possible value for all other columns, and seek to that.
  //    If this matches, this is the lower bound of our desired scan.
  // 3. If we found our desired lower bound, find an upper bound for the scan
  //    by searching for the next row key matching one value higher than the
  //    highest value that will match our predicate.
  //
  // We track two pointers with skip-scan:
  //   - the primary key (PK) value index (ad-hoc index) search pointer,
  //     which gives us a forward index of value to position
  //   - the scan pointer, which is cur_idx_, representing an offset one more
  //     than the last row that we actually scanned (in the previous batch)
  //
  // Currently lookup by position (ordinal) is not supported for ad-hoc index (value based index),
  // due to this, in some cases PK lookups may land us at an offset that we had already scanned.
  // In this case, we continue with the PK lookups until the upper bound key offset
  // (skip_scan_upper_bound_idx_) <= cur_idx_.

  skip_scan_upper_bound_idx_ = upper_bound_idx_;
  size_t skip_scan_lower_bound_idx = cur_idx_;
  // Whether we found our lower bound key.
  bool lower_bound_key_found = false;

  // Continue seeking the next matching row if we didn't find
  // a predicate match.
  for (int loop_num = 0;
       !lower_bound_key_found && loop_num < FLAGS_skip_scan_short_circuit_loops;
       loop_num++) {
    DCHECK_LT(cur_idx_, skip_scan_upper_bound_idx_);

    // Step 1. search for the next distinct prefix.
    Status s;
    // We only want to seek to the first entry if this is the first time we
    // are entering this loop on the first call to this method.
    if (cur_idx_ == 0 && loop_num == 0) {
      // Get the first entry of the validx_iter_.
      s = key_iter_->SeekToFirst();
    } else if (skip_scan_searched_cur_prefix_) { // Only seek to the next prefix if
                                                 // our previous call to
                                                 // SeekToRowWithCurPrefixMatchingPred()
                                                 // didn't "roll" past the previous prefix.
      s = SeekToNextPrefixKey(skip_scan_predicate_column_id_, /* cache_seeked_value=*/ true);

      skip_scan_num_seeks_++;
      // Disable skip scan on the fly (see the .h file for details).
      if (skip_scan_num_seeks_ >= skip_scan_num_seeks_cutoff_) {
        use_skip_scan_ = false;
        VLOG(1) << strings::Substitute("Disabled index skip scan. Number of seeks = $0. Current"
                                       " row index = $1", skip_scan_num_seeks_, cur_idx_);
        return Status::OK();
      }
    }
    skip_scan_searched_cur_prefix_ = true;

    // We fell off the end of the cfile. No more rows will match.
    if (s.IsNotFound()) {
      lower_bound_key_found = false;
      break;
    }
    RETURN_NOT_OK(s);

    // Step 2. seek to the lower bound of our desired scan.

    // Clear the buffer that stores the encoded key.
    gscoped_ptr<EncodedKey> next_prefix_key;
    RETURN_NOT_OK(DecodeCurrentKey(&next_prefix_key));
    // Attempt to seek to the row with predicate match.
    s = SeekToRowWithCurPrefixMatchingPred(next_prefix_key);
    if (s.IsNotFound()) {
      lower_bound_key_found = false;
      break;
    }
    RETURN_NOT_OK(s);

    gscoped_ptr<EncodedKey> lower_bound_key;
    // Check if we successfully seeked to a predicate key match.
    RETURN_NOT_OK(DecodeCurrentKey(&lower_bound_key));
    // Keep track of the lower bound on a matching key.
    skip_scan_lower_bound_idx = key_iter_->GetCurrentOrdinal();
    // Does this lower bound key match ?
    // This check is only for the predicate column value match.
    // Even if the prefix key does not match, skip scan flow will work
    // as expected.
    lower_bound_key_found = CheckPredicateMatch(lower_bound_key);
    // We weren't able to find a predicate match for our lower bound key, so loop and search again.
    if (!lower_bound_key_found) {
      // If the prefix key rolled between our initial lower bound next prefix
      // seek and our seek to the predicate match with that prefix, it's
      // possible that the latest prefix will have a predicate match, so on our
      // next iteration of the loop, we should not seek to the next prefix.
      //
      // For eg :
      // consider the following two tables with columns P1, P2 and P3.
      // note: P1 and P2 are key columns and table rows are sorted by the key columns.
      // predicate: P2 = "3".
      //
      // TABLE 1.
      // P1   P2    P3
      // 1     2     1
      // 1     4     1    <------  key iterator position after calling
      // 2     3     1             SeekToRowWithCurPrefixMatchingPred(..) with prefix key = "1".
      //
      // TABLE 2.
      // P1   P2    P3
      // 1     2     1
      // 2     1     1    <------  key iterator position after calling
      // 2     3     1             SeekToRowWithCurPrefixMatchingPred(..) with prefix key = "1".
      //
      // Note that in both the cases above, the predicate column value ("3") is not present
      // in the row pointed to by the iterator. Here, an important point to note is that in
      // case of Table 2. the prefix value has already rolled over to the next prefix key ("2").
      // So, to avoid continuing to seek to the next prefix key in the next loop iteration we set
      // skip_scan_searched_cur_prefix_ to false. This means that we have not yet searched the
      // latest prefix key ("2") for the predicate match, hence no need to continue seeking for
      // the next prefix key in the next loop iteration.
      //
      if (!KeyColumnsMatch(next_prefix_key, lower_bound_key,
          /* start_col_id= */ 0,
          /* end_col_id= */ skip_scan_predicate_column_id_ - 1)) {
        skip_scan_searched_cur_prefix_ = false;
      }
      continue;
    }

    // Step 3. seek to the upper bound of our desired scan.
    // TODO(anupama): to support in-range predicates, seek right after the
    // last occurrence of the row containing the prefix key and predicate column
    // containing the upper bound of the predicate values.

    // Note: To handle a similar situation (as illustrated with Tables above) when finding the
    // upper bound key offset, we follow a different approach. We simply do not cache
    // the seeked value for the upper bound key, hence cache_seeked_value = false below.
    // However, in this case the 'cache_seeked_value' parameter has additional
    // semantics: not only are we not caching the seeked value, but we're
    // searching for a different value than if `cache_seeked_value` were true.
    s = SeekToNextPrefixKey(skip_scan_predicate_column_id_, /* cache_seeked_value=*/ false);
    if (s.IsNotFound()) {
      // We hit the end of the file. Simply scan to the end.
      skip_scan_upper_bound_idx_ = upper_bound_idx_;
      break;
    }
    RETURN_NOT_OK(s);

    skip_scan_upper_bound_idx_ = key_iter_->GetCurrentOrdinal();
    // Check to see whether we have effectively seeked backwards. If so, we
    // need to keep looking until our upper bound is past the last row that we
    // previously scanned.
    if (skip_scan_upper_bound_idx_ <= cur_idx_) {
      skip_scan_upper_bound_idx_ = upper_bound_idx_; // Reset upper bound to max.
      lower_bound_key_found = false;
    }
  }

  // Now update cur_idx_, which controls the next row we will scan.
  // Seek to the next lower bound match.
  // Never seek backward. For details, refer to the comment about tracking two
  // pointers with skip-scan (near the method beginning).
  cur_idx_ = std::max<int64_t>(cur_idx_, skip_scan_lower_bound_idx);
  if (!lower_bound_key_found) {
    // TODO(anupama): We scan a single row (guaranteed not to match) for now, because
    // having entered PrepareBatch() implies that we have rows to scan. Can
    // we prepare 0 rows instead of doing this?
    *remaining = 1;
    // Reset our upper bound since we use it to short-circuit.
    skip_scan_upper_bound_idx_ = -1;
  } else {
    // Always read at least one row.
    *remaining = std::max<int64_t>(skip_scan_upper_bound_idx_ - cur_idx_, 1);
  }
  return Status::OK();
}

Status CFileSet::Iterator::PrepareBatch(size_t *nrows) {
  DCHECK_EQ(prepared_count_, 0) << "Already prepared";
  size_t remaining = upper_bound_idx_ - cur_idx_;
  if (use_skip_scan_) {
    Status s = SkipToNextScan(&remaining);
    if (!s.ok()) {
      LOG(WARNING) << "Skip scan failed: " << s.ToString();
      use_skip_scan_ = false;
    }
  }

  if (*nrows > remaining) {
    *nrows = remaining;
  }
  prepared_count_ = *nrows;
  // Lazily prepare the first column when it is materialized.
  return Status::OK();
}

Status CFileSet::Iterator::PrepareColumn(ColumnMaterializationContext *ctx) {
  ColumnIterator* col_iter = col_iters_[ctx->col_idx()].get();
  size_t n = prepared_count_;
  if (!col_iter->seeked() || col_iter->GetCurrentOrdinal() != cur_idx_) {
    // Either this column has not yet been accessed, or it was accessed
    // but then skipped in a prior block (e.g because predicates on other
    // columns completely eliminated the block).
    //
    // Either way, we need to seek it to the correct offset.
    std::cout << "wangxixu-seektoordinal-column_id:" << ctx->col_idx() << std::endl;
    RETURN_NOT_OK(col_iter->SeekToOrdinal(cur_idx_));
  }
  std::cout << "wangxixu-prepare-column_id:" << ctx->col_idx() << std::endl;
  Status s = col_iter->PrepareBatch(&n);
  if (!s.ok()) {
    LOG(WARNING) << "Unable to prepare column " << ctx->col_idx() << ": " << s.ToString();
    return s;
  }

  if (n != prepared_count_) {
    return Status::Corruption(
            StringPrintf("Column %zd (%s) didn't yield enough rows at offset %zd: expected "
                                 "%zd but only got %zd", ctx->col_idx(),
                         projection_->column(ctx->col_idx()).ToString().c_str(),
                         cur_idx_, prepared_count_, n));
  }

  prepared_iters_.emplace_back(col_iter);

  return Status::OK();
}

Status CFileSet::Iterator::InitializeSelectionVector(SelectionVector *sel_vec) {
  sel_vec->SetAllTrue();
  return Status::OK();
}

Status CFileSet::Iterator::MaterializeColumn(ColumnMaterializationContext *ctx) {
  CHECK_EQ(prepared_count_, ctx->block()->nrows());
  DCHECK_LT(ctx->col_idx(), col_iters_.size());
  RETURN_NOT_OK(PrepareColumn(ctx));
  ColumnIterator* iter = col_iters_[ctx->col_idx()].get();
  RETURN_NOT_OK(iter->Scan(ctx));

  return Status::OK();
}

Status CFileSet::Iterator::FinishBatch() {
  DCHECK_GT(prepared_count_, 0);

  for (ColumnIterator* col_iter : prepared_iters_) {
    RETURN_NOT_OK(col_iter->FinishBatch());
  }

  cur_idx_ += prepared_count_;
  Unprepare();

  return Status::OK();
}


void CFileSet::Iterator::GetIteratorStats(vector<IteratorStats>* stats) const {
  stats->clear();
  stats->reserve(col_iters_.size());
  for (const auto& iter : col_iters_) {
    ANNOTATE_IGNORE_READS_BEGIN();
    stats->push_back(iter->io_statistics());
    ANNOTATE_IGNORE_READS_END();
  }
}

} // namespace tablet
} // namespace kudu
