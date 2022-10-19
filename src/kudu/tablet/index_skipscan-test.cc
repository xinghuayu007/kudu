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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "kudu/common/column_predicate.h"
#include "kudu/common/common.pb.h"
#include "kudu/common/generic_iterators.h"
#include "kudu/common/iterator.h"
#include "kudu/common/partial_row.h"
#include "kudu/common/row.h"
#include "kudu/common/scan_spec.h"
#include "kudu/common/schema.h"
#include "kudu/gutil/stringprintf.h"
#include "kudu/tablet/cfile_set.h"
#include "kudu/tablet/diskrowset.h"
#include "kudu/tablet/local_tablet_writer.h"
#include "kudu/tablet/tablet-test-util.h"
#include "kudu/tablet/tablet.h"
#include "kudu/tablet/tablet_metadata.h"
#include "kudu/util/bloom_filter.h"
#include "kudu/util/mem_tracker.h"
#include "kudu/util/random.h"
#include "kudu/util/slice.h"
#include "kudu/util/status.h"
#include "kudu/util/test_macros.h"
#include "kudu/util/test_util.h"

using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

DECLARE_bool(enable_skip_scan);
DECLARE_int32(max_encoded_key_size_bytes);

namespace kudu {
namespace tablet {

class RowSetMetadata;

// Schemas vary in the number and types of composite primary keys.
enum SchemaType {
  kOnePK,
  kTwoPK,
  kThreePK,
  kFourPK,
  kFivePK,
  kTwoPKrandom,
  kThreePKrandom,
  kFourPKrandom,
  kFivePKrandom,
  kMaxSizedEncodedKey,
  kMiddleColumnMinKeyValue,
  kMiddleColumnMaxKeyValue,
  kMiddleColumnCustom
};

const SchemaType kSchemaTypesList[] = {
  kOnePK,
  kTwoPK,
  kThreePK,
  kFourPK,
  kFivePK,
  kTwoPKrandom,
  kThreePKrandom,
  kFourPKrandom,
  kFivePKrandom,
  kMaxSizedEncodedKey,
  kMiddleColumnMinKeyValue,
  kMiddleColumnMaxKeyValue,
  kMiddleColumnCustom
};

class IndexSkipScanTest : public KuduTabletTest,
                          public ::testing::WithParamInterface<std::tuple<SchemaType, bool>> {
 public:

  enum DataGeneratorType {
    ALL_VALUES_RANDOM,
    FIXED_PRED_COLUMN_VALUE_FOR_HALF,
  };

  SchemaType schema_type;
  std::shared_ptr<RowSetMetadata> rowset_meta_;

  IndexSkipScanTest()
      : KuduTabletTest(CreateSchema(std::get<0>(GetParam()))) {
    const auto& param = GetParam();
    schema_type = std::get<0>(param);
    FLAGS_enable_skip_scan = std::get<1>(param);
  }

  void SetUp() override {
    KuduTabletTest::SetUp();
    ASSERT_OK(tablet()->metadata()->CreateRowSet(&rowset_meta_));
    FillTestTablet();
  }

  // Generate and insert given number of rows using the given PRNG,
  // return # rows generated that match the predicate_val on predicate_col.
  // The pattern for the predicate column is specified by 'gen_type' parameter,
  // all other columns are populated with random values.
  int GenerateData(Random* random,
                   int num_rows,
                   int predicate_col_id,
                   int64_t predicate_value,
                   DataGeneratorType gen_type) {
    LocalTabletWriter writer(tablet().get(), &client_schema_);
    KuduPartialRow row(&client_schema_);

    const size_t num_key_cols = client_schema_.num_key_columns();
    int num_matching = 0;

    while (num_rows > 0) {
      bool matched = false;
      for (int col_idx = 0; col_idx < num_key_cols; col_idx++) {
        if (gen_type == ALL_VALUES_RANDOM) {
          int64_t value = random->Uniform(1000);
          CHECK_OK(row.SetInt64(col_idx, value));
          if (col_idx == predicate_col_id && value == predicate_value) {
            matched = true;
          }
        } else {
          CHECK(gen_type == FIXED_PRED_COLUMN_VALUE_FOR_HALF);
          if (col_idx == predicate_col_id) {
            int64_t value;
            switch (num_rows % 2) {
              case 0:
                value = predicate_value;
                break;
              case 1:
                value = random->Uniform(100);
                break;
            }
            matched = (value == predicate_value);
            CHECK_OK(row.SetInt64(col_idx, value));
            continue;
          }

          int64_t value = random->Uniform(100);
          CHECK_OK(row.SetInt64(col_idx, value));
        }
      }
      Status s =  writer.Insert(row);
      // As keys are inserted randomly, retry row insertion in case
      // the current row insertion failed due to duplicate value.
      if (s.IsAlreadyPresent()) {
        continue;
      }
      CHECK_OK(s);

      if (matched) {
        ++num_matching;
      }
      --num_rows;
    }
    CHECK_OK(tablet()->Flush());
    return num_matching;
  }

  Schema CreateSchema(SchemaType schema_type) {
    SchemaBuilder builder;
    switch (schema_type) {
      case kOnePK:
        CHECK_OK(builder.AddKeyColumn("P1", INT32));
        break;

      case kTwoPK:
        CHECK_OK(builder.AddKeyColumn("P1", INT32));
        CHECK_OK(builder.AddKeyColumn("P2", INT16));
        break;

      case kThreePK:
        CHECK_OK(builder.AddKeyColumn("P1", INT32));
        CHECK_OK(builder.AddKeyColumn("P2", INT16));
        CHECK_OK(builder.AddKeyColumn("P3", STRING));
        break;

      case kFourPK:
        CHECK_OK(builder.AddKeyColumn("P1", INT32));
        CHECK_OK(builder.AddKeyColumn("P2", INT16));
        CHECK_OK(builder.AddKeyColumn("P3", STRING));
        CHECK_OK(builder.AddKeyColumn("P4", INT8));
        break;

      case kFivePK:
        CHECK_OK(builder.AddKeyColumn("P1", INT32));
        CHECK_OK(builder.AddKeyColumn("P2", INT16));
        CHECK_OK(builder.AddKeyColumn("P3", STRING));
        CHECK_OK(builder.AddKeyColumn("P4", INT8));
        CHECK_OK(builder.AddKeyColumn("P5", INT8));
        break;

      case kTwoPKrandom:
        CHECK_OK(builder.AddKeyColumn("P1", INT64));
        CHECK_OK(builder.AddKeyColumn("P2", INT64));
        break;

      case kFourPKrandom:
        CHECK_OK(builder.AddKeyColumn("P1", INT64));
        CHECK_OK(builder.AddKeyColumn("P2", INT64));
        CHECK_OK(builder.AddKeyColumn("P3", INT64));
        CHECK_OK(builder.AddKeyColumn("P4", INT64));
        break;

      case kFivePKrandom:
        CHECK_OK(builder.AddKeyColumn("P1", INT64));
        CHECK_OK(builder.AddKeyColumn("P2", INT64));
        CHECK_OK(builder.AddKeyColumn("P3", INT64));
        CHECK_OK(builder.AddKeyColumn("P4", INT64));
        CHECK_OK(builder.AddKeyColumn("P5", INT64));
        break;

      case kMaxSizedEncodedKey:
        CHECK_OK(builder.AddKeyColumn("P1", STRING));
        CHECK_OK(builder.AddKeyColumn("P2", STRING));
        break;

      case kThreePKrandom:
      case kMiddleColumnMinKeyValue:
      case kMiddleColumnMaxKeyValue:
      case kMiddleColumnCustom:
        CHECK_OK(builder.AddKeyColumn("P1", INT64));
        CHECK_OK(builder.AddKeyColumn("P2", INT64));
        CHECK_OK(builder.AddKeyColumn("P3", INT64));
        break;
    }
    return builder.BuildWithoutIds();
  }

  void FillTestTablet() {

    DiskRowSetWriter rsw(rowset_meta_.get(), &schema_,
                         BloomFilterSizing::BySizeAndFPRate(32*1024, 0.01f));
    ASSERT_OK(rsw.Open());
    RowBuilder rb(&client_schema_);

    constexpr int32_t kNumDistinctP1 = 2;
    constexpr int16_t kNumDistinctP2 = 10;
    constexpr int kNumDistinctP3 = 5;
    constexpr int8_t kNumDistinctP4 = 1;
    constexpr int8_t kNumDistinctP5 = 20;

    switch (schema_type) {
      case kOnePK: {
        for (int32_t p1 = 1; p1 <= kNumDistinctP1; p1++) {
          rb.Reset();
          rb.AddInt32(p1);
          ASSERT_OK(WriteRow(rb.data(), &rsw));
        }
        break;
      }

      case kTwoPK: {
        for (int32_t p1 = 1; p1 <= kNumDistinctP1; p1++) {
          for (int16_t p2 = 1; p2 <= kNumDistinctP2; p2++) {
            rb.Reset();
            rb.AddInt32(p1);
            rb.AddInt16(p2);
            ASSERT_OK(WriteRow(rb.data(), &rsw));
          }
        }
        break;
      }

      case kThreePK: {
        for (int32_t p1 = 1; p1 <= kNumDistinctP1; p1++) {
          for (int16_t p2 = 1; p2 <= kNumDistinctP2; p2++) {
            for (int p3 = 1; p3 <= kNumDistinctP3; p3++) {
              rb.Reset();
              rb.AddInt32(p1);
              rb.AddInt16(p2);
              rb.AddString(StringPrintf("%d_p3", p3));
              ASSERT_OK(WriteRow(rb.data(), &rsw));
            }
          }
        }
        break;
      }

      case kFourPK: {
        for (int32_t p1 = 1; p1 <= kNumDistinctP1; p1++) {
          for (int16_t p2 = 1; p2 <= kNumDistinctP2; p2++) {
            for (int p3 = 1; p3 <= kNumDistinctP3; p3++) {
              for (int8_t p4 = 1; p4 <= kNumDistinctP4; p4++) {
                rb.Reset();
                rb.AddInt32(p1);
                rb.AddInt16(p2);
                rb.AddString(StringPrintf("%d_p3", p3));
                rb.AddInt8(p4);
                ASSERT_OK(WriteRow(rb.data(), &rsw));
              }
            }
          }
        }
        break;
      }

      case kFivePK: {
        for (int32_t p1 = 1; p1 <= kNumDistinctP1; p1++) {
          for (int16_t p2 = 1; p2 <= kNumDistinctP2; p2++) {
            for (int p3 = 1; p3 <= kNumDistinctP3; p3++) {
              for (int8_t p4 = 1; p4 <= kNumDistinctP4; p4++) {
                for (int8_t p5 = 1; p5 <= kNumDistinctP5; p5++) {
                  rb.Reset();
                  rb.AddInt32(p1);
                  rb.AddInt16(p2);
                  rb.AddString(StringPrintf("%d_p3", p3));
                  rb.AddInt8(p4);
                  rb.AddInt8(p5);
                  ASSERT_OK(WriteRow(rb.data(), &rsw));
                }
              }
            }
          }
        }
        break;
      }

      case kMaxSizedEncodedKey: {
        // Check for the case when the encoded key size  = max allowable size for encoded key
        rb.AddString(string(FLAGS_max_encoded_key_size_bytes / 2, 'a'));
        rb.AddString(string(FLAGS_max_encoded_key_size_bytes / 2, 'b'));
        ASSERT_OK(WriteRow(rb.data(), &rsw));
        break;
      }

      case kMiddleColumnCustom: {
        const int64_t predicate_val = 0;

        rb.AddInt64(std::numeric_limits<int64_t>::min());
        rb.AddInt64(predicate_val);
        rb.AddInt64(std::numeric_limits<int64_t>::max());
        ASSERT_OK(WriteRow(rb.data(), &rsw));

        rb.Reset();
        rb.AddInt64(0);
        rb.AddInt64(predicate_val);
        rb.AddInt64(0);
        ASSERT_OK(WriteRow(rb.data(), &rsw));

        rb.Reset();
        rb.AddInt64(std::numeric_limits<int64_t>::max());
        rb.AddInt64(predicate_val);
        rb.AddInt64(std::numeric_limits<int64_t>::min());
        ASSERT_OK(WriteRow(rb.data(), &rsw));
        break;
      }

      default: {
        return;
      }
    }
    ASSERT_OK(rsw.Finish());
    ASSERT_OK(tablet()->Flush());
  }

  void ScanTablet(ScanSpec* spec, vector<string>* results,
                  const char* descr, bool skip_scan_expected_flag) {
    SCOPED_TRACE(descr);

    shared_ptr<CFileSet> fileset;
    ASSERT_OK(CFileSet::Open(rowset_meta_, MemTracker::GetRootTracker(),
                             MemTracker::GetRootTracker(), nullptr, &fileset));
    unique_ptr<CFileSet::Iterator> cfile_iter(fileset->NewIterator(&schema_, nullptr));
    // The unique_ptr is move()d and we want to be able to assert on some
    // internals later.
    CFileSet::Iterator* raw_cfile_iter = cfile_iter.get();

    unique_ptr<RowwiseIterator> iter(NewMaterializingIterator(std::move(cfile_iter)));

    ASSERT_OK(iter->Init(spec));

    if (FLAGS_enable_skip_scan) {
      // Check if use_skip_scan_ is set to the expected value.
      ASSERT_EQ(raw_cfile_iter->use_skip_scan_, skip_scan_expected_flag);
    } else {
      ASSERT_TRUE(!raw_cfile_iter->use_skip_scan_);
    }

    ASSERT_TRUE(spec->predicates().empty()) << "Should have accepted all predicates";
    ASSERT_OK(IterateToStringList(iter.get(), results));
  }

  void ScanTabletForRandomCases(ScanSpec *spec, vector<string> *results, const char *descr) {
    SCOPED_TRACE(descr);
    unique_ptr<RowwiseIterator> iter;
    ASSERT_OK(tablet()->NewRowIterator(client_schema_, &iter));
    ASSERT_OK(iter->Init(spec));
    ASSERT_TRUE(spec->predicates().empty()) << "Should have accepted all predicates";
    ASSERT_OK(IterateToStringList(iter.get(), results));
  }
};

// The following set of tests evaluate the scan results with different schema types.
// This is mainly done to verify the correctness of index skip scan approach.
TEST_P(IndexSkipScanTest, IndexSkipScanCorrectnessTest) {
  Random random(SeedRandom());
  LOG(INFO) << "wangxixu-schema-type:" << schema_type;
  switch (schema_type) {
    case kOnePK: {
      // Test predicate on the PK column.
      ScanSpec spec;
      int32_t value_p1 = 2;
      auto pred_p1 = ColumnPredicate::Equality(schema_.column(0), &value_p1);
      spec.AddPredicate(pred_p1);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "Exact match on column P1", false));
      EXPECT_EQ(1, results.size());
      break;
    }

    case kTwoPK: {
      // Test predicate on the first PK column.
    //   ScanSpec spec;
    //   int32_t value_p1 = 2;
    //   auto pred_p1 = ColumnPredicate::Equality(schema_.column(0), &value_p1);
    //   spec.AddPredicate(pred_p1);
    //   LOG(INFO) << "wangxiux-test-on-first-pk: " << pred_p1.ToString();
    //   vector<string> results;
    //   NO_FATALS(ScanTablet(&spec, &results, "Exact match on P1", false));
    //   EXPECT_EQ(10, results.size());
      }
      {
        // Test predicate on the second PK column.
        ScanSpec spec;
        int16_t value_p2 = 9;
        auto pred_p2 = ColumnPredicate::Equality(schema_.column(1), &value_p2);
        spec.AddPredicate(pred_p2);
        LOG(INFO) << "wangxiux-test-on-second-pk: " << pred_p2.ToString();
        vector<string> results;
        NO_FATALS(ScanTablet(&spec, &results, "Exact match on P2", true));
        EXPECT_EQ(2, results.size());
      }
      {
        // Test predicate on the first and second PK column.
        // ScanSpec spec;
        // int32_t value_p1 = 1;
        // int16_t value_p2 = 1;
        // auto pred_p1 = ColumnPredicate::Equality(schema_.column(0), &value_p1);
        // auto pred_p2 = ColumnPredicate::Equality(schema_.column(1), &value_p2);
        // spec.AddPredicate(pred_p1);
        // spec.AddPredicate(pred_p2);
        // vector<string> results;
        // NO_FATALS(ScanTablet(&spec, &results, "Exact match on P1 and P2", false));
        // EXPECT_EQ(1, results.size());
      }
      break;

    case kThreePK: {
      // Test predicate on the third PK column.
      ScanSpec spec;
      Slice value_p3("2_p3");
      auto pred_p3 = ColumnPredicate::Equality(schema_.column(2), &value_p3);
      spec.AddPredicate(pred_p3);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "Exact match on P3", true));
      EXPECT_EQ(20, results.size());
      break;
    }

    case kFourPK: {
      // Test predicate on the fourth PK column on a non-existent value.
      ScanSpec spec;
      int16_t value_p4 = 3;
      auto pred_p4 = ColumnPredicate::Equality(schema_.column(3), &value_p4);
      spec.AddPredicate(pred_p4);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "Exact match on P4", true));
      EXPECT_EQ(0, results.size());
    }
    {
      // Test predicate on the fourth PK column.
      ScanSpec spec;
      int16_t p4 = 1;
      auto pred_p1 = ColumnPredicate::Equality(schema_.column(3), &p4);
      spec.AddPredicate(pred_p1);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "Exact match on P4", true));
      EXPECT_EQ(100, results.size());
    }
      break;

    case kFivePK: {
      // Test predicate on the fifth PK column.
      ScanSpec spec;
      int16_t value_p5 = 20;
      auto pred_p5 = ColumnPredicate::Equality(schema_.column(4), &value_p5);
      spec.AddPredicate(pred_p5);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "Exact match on P5", true));
      EXPECT_EQ(100, results.size());
    }
    {
      // Test predicate on the third and fifth PK column.
      ScanSpec spec;
      Slice value_p3("5_p3");
      int16_t value_p5 = 20;
      auto pred_p3 = ColumnPredicate::Equality(schema_.column(2), &value_p3);
      auto pred_p5 = ColumnPredicate::Equality(schema_.column(4), &value_p5);
      spec.AddPredicate(pred_p3);
      spec.AddPredicate(pred_p5);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "Exact match on P3 and P5", true));
      EXPECT_EQ(20, results.size());
    }
      break;

    // The following tests scan results on random data,
    // where each key value is in the range [0..1000].
    case kTwoPKrandom: {
      const int num_rows = 100;
      int64_t predicate_val = random.Uniform(1000);
      int predicate_col_id = 1;
      int num_matching = GenerateData(&random, num_rows,
                                      predicate_col_id, predicate_val,
                                      ALL_VALUES_RANDOM);

      ScanSpec spec;
      auto pred = ColumnPredicate::Equality(schema_.column(predicate_col_id), &predicate_val);
      spec.AddPredicate(pred);
      vector<string> results;
      NO_FATALS(ScanTabletForRandomCases(&spec, &results, "Exact match on P2"));
      ASSERT_EQ(num_matching, results.size());
      break;
    }

    case kThreePKrandom: {
      const int num_rows = 1000;
      int64_t predicate_val = random.Uniform(1000);
      int predicate_col_id = 1;
      int num_matching = GenerateData(&random, num_rows,
                                      predicate_col_id, predicate_val,
                                      ALL_VALUES_RANDOM);

      ScanSpec spec;
      auto pred = ColumnPredicate::Equality(schema_.column(predicate_col_id), &predicate_val);
      spec.AddPredicate(pred);
      vector<string> results;
      NO_FATALS(ScanTabletForRandomCases(&spec, &results, "Exact match on P2"));
      ASSERT_EQ(num_matching, results.size());
      break;
    }

    case kFourPKrandom: {
      const int num_rows = 1000;
      int64_t predicate_val = random.Uniform(1000);
      int predicate_col_id = 1;
      int num_matching = GenerateData(&random, num_rows,
                                      predicate_col_id, predicate_val,
                                      ALL_VALUES_RANDOM);

      ScanSpec spec;
      auto pred = ColumnPredicate::Equality(schema_.column(predicate_col_id), &predicate_val);
      spec.AddPredicate(pred);
      vector<string> results;
      NO_FATALS(ScanTabletForRandomCases(&spec, &results, "Exact match on P2"));
      ASSERT_EQ(num_matching, results.size());
      break;
    }

    case kFivePKrandom: {
      const int num_rows = 1000;
      int64_t predicate_val = random.Uniform(1000);
      int predicate_col_id = 3;
      int num_matching = GenerateData(&random, num_rows,
                                      predicate_col_id, predicate_val,
                                      ALL_VALUES_RANDOM);

      ScanSpec spec;
      auto pred = ColumnPredicate::Equality(schema_.column(predicate_col_id), &predicate_val);
      spec.AddPredicate(pred);
      vector<string> results;
      NO_FATALS(ScanTabletForRandomCases(&spec, &results, "Exact match on P4"));
      ASSERT_EQ(num_matching, results.size());
      break;
    }

    case kMaxSizedEncodedKey: {
      ScanSpec spec;
      string str_value(FLAGS_max_encoded_key_size_bytes / 2, 'b');
      Slice value_p2(str_value);
      auto pred_p2 = ColumnPredicate::Equality(schema_.column(1), &value_p2);
      spec.AddPredicate(pred_p2);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "Exact match on P2", true));
      EXPECT_EQ(1, results.size());
      break;
    }

    case kMiddleColumnMinKeyValue: {
      const int num_rows = 25;
      constexpr int64_t predicate_val = std::numeric_limits<int64_t>::min();
      constexpr int predicate_col_id = 1;
      const int num_matching = GenerateData(&random, num_rows,
                                            predicate_col_id, predicate_val,
                                            FIXED_PRED_COLUMN_VALUE_FOR_HALF);
      ScanSpec spec;
      auto pred = ColumnPredicate::Equality(schema_.column(predicate_col_id), &predicate_val);
      spec.AddPredicate(pred);
      vector<string> results;
      NO_FATALS(ScanTabletForRandomCases(&spec, &results, "exact match on P2"));
      ASSERT_EQ(num_matching, results.size());
      break;
    }

    case kMiddleColumnMaxKeyValue: {
      const int num_rows = 50;
      constexpr int64_t predicate_val = std::numeric_limits<int64_t>::max();
      constexpr int predicate_col_id = 1;
      const int num_matching = GenerateData(&random, num_rows,
                                            predicate_col_id, predicate_val,
                                            FIXED_PRED_COLUMN_VALUE_FOR_HALF);
      ScanSpec spec;
      auto pred = ColumnPredicate::Equality(schema_.column(predicate_col_id), &predicate_val);
      spec.AddPredicate(pred);
      vector<string> results;
      NO_FATALS(ScanTabletForRandomCases(&spec, &results, "exact match on P2"));
      ASSERT_EQ(num_matching, results.size());
      break;
    }

    case kMiddleColumnCustom: {
      int64_t predicate_val = 0;
      ScanSpec spec;
      auto pred = ColumnPredicate::Equality(schema_.column(1), &predicate_val);
      spec.AddPredicate(pred);
      vector<string> results;
      NO_FATALS(ScanTablet(&spec, &results, "exact match on P2", true));
      ASSERT_EQ(3, results.size());
      break;
    }

  }
}
INSTANTIATE_TEST_CASE_P(IndexSkipScanCorrectnessTest, IndexSkipScanTest,
                        ::testing::Combine(::testing::ValuesIn(kSchemaTypesList),
                                           ::testing::Bool()));

} // namespace tablet
} // namespace kudu
