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

#include "kudu/master/txn_manager.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "kudu/common/wire_protocol.h"
#include "kudu/gutil/port.h"
#include "kudu/master/master.h"
#include "kudu/master/txn_manager.pb.h"
#include "kudu/transactions/txn_system_client.h"
#include "kudu/util/flag_tags.h"
#include "kudu/util/monotime.h"
#include "kudu/util/net/net_util.h"

using kudu::master::Master;
using kudu::transactions::TxnStatusEntryPB;
using std::string;
using std::vector;

// TODO(aserbin): remove the flag once the txn-related work is complete.
DEFINE_bool(txn_manager_enabled, false,
            "Whether to enable TxnManager (this enables a new feature)");
TAG_FLAG(txn_manager_enabled, hidden);
TAG_FLAG(txn_manager_enabled, experimental);

// For Kudu clusters which use transactions we should prefer non-lazy
// initialization of the TxnManager. That's to reduce the latency of the very
// first call processed by the TxnManager after it starts.
//
// TODO(aserbin): maybe, consider changing the default value of this flag to
//                'false' once proper tablet type filtering is implemented or
//                --txn_manager_lazily_initialized=true is added for existing
//                test scenarios which assume there isn't a single table
//                in the cluster at the time they start.
DEFINE_bool(txn_manager_lazily_initialized, true,
            "Whether to initialize TxnManager upon arrival of first request. "
            "Otherwise, TxnManager is initialized upon master's startup.");
TAG_FLAG(txn_manager_lazily_initialized, advanced);
TAG_FLAG(txn_manager_lazily_initialized, experimental);

// TODO(aserbin): clarify on the proper value for the span of the transaction
//                status table's range partition. At this point it's not yet
//                crystal clear what criteria are essential when defining one.
DEFINE_int64(txn_manager_status_table_range_partition_span, 1000000,
             "A span for a status table's range partition. Once TxnManager "
             "detects there isn't a backing tablet for a transaction "
             "identifier, it adds a new range partition in the transaction "
             "status table with the lower bound equal to the upper bound of "
             "the previous range and (lower bound + span) as the upper bound.");
TAG_FLAG(txn_manager_status_table_range_partition_span, advanced);
TAG_FLAG(txn_manager_status_table_range_partition_span, experimental);

namespace kudu {
namespace transactions {

namespace {

// If 's' is not OK and 'resp' has no application specific error set,
// set the error field of 'resp' to match 's' and set the code to
// UNKNOWN_ERROR.
template<class RespClass>
void CheckRespErrorOrSetUnknown(const Status& s, RespClass* resp) {
  if (!s.ok() && !resp->has_error()) {
    StatusToPB(s, resp->mutable_error()->mutable_status());
    resp->mutable_error()->set_code(TxnManagerErrorPB::UNKNOWN_ERROR);
  }
}

// Conversion of a deadline specified for an RPC into a timeout, i.e.
// convert a point in time to a delta between current time and the specified
// point in time.
MonoDelta ToDelta(const MonoTime& deadline) {
  MonoDelta timeout = deadline == MonoTime::Max()
        ? MonoDelta::FromNanoseconds(std::numeric_limits<int64_t>::max())
        : deadline - MonoTime::Now();
  return timeout;
}

} // anonymous namespace

TxnManager::TxnManager(Master* server)
    : is_lazily_initialized_(FLAGS_txn_manager_lazily_initialized),
      server_(server),
      need_init_(true),
      initialized_(false),
      next_txn_id_(0) {
}

TxnManager::~TxnManager() {
}

Status TxnManager::BeginTransaction(const string& /* username */,
                                    const MonoTime& deadline,
                                    int64_t* txn_id,
                                    int32_t* keep_alive_interval_ms) {
  DCHECK(txn_id);
  DCHECK(keep_alive_interval_ms);
  RETURN_NOT_OK(CheckInitialized(deadline));
  // TODO(aserbin): this is implemented in a follow-up changelist
  return Status::NotSupported("BeginTransaction is not supported yet");
}

Status TxnManager::CommitTransaction(int64_t txn_id,
                                     const string& username,
                                     const MonoTime& deadline) {
  RETURN_NOT_OK(CheckInitialized(deadline));
  return txn_sys_client_->BeginCommitTransaction(
      txn_id, username, ToDelta(deadline));
}

Status TxnManager::GetTransactionState(int64_t txn_id,
                                       const string& username,
                                       const MonoTime& deadline,
                                       TxnStatusEntryPB* txn_status) {
  DCHECK(txn_status);
  RETURN_NOT_OK(CheckInitialized(deadline));
  return txn_sys_client_->GetTransactionStatus(
      txn_id, username, txn_status, ToDelta(deadline));
}

Status TxnManager::AbortTransaction(int64_t txn_id,
                                    const string& username,
                                    const MonoTime& deadline) {
  RETURN_NOT_OK(CheckInitialized(deadline));
  return txn_sys_client_->AbortTransaction(txn_id, username, ToDelta(deadline));
}

Status TxnManager::KeepTransactionAlive(int64_t /* txn_id */,
                                        const string& /* username */,
                                        const MonoTime& deadline) {
  RETURN_NOT_OK(CheckInitialized(deadline));
  // TODO(aserbin): call txn_sys_client_ once the functionality is there
  return Status::NotSupported("KeepTransactionAlive is not supported yet");
}

// This method isn't supposed to be called concurrently, so there isn't any
// protection against concurrent calls.
Status TxnManager::Init() {
  DCHECK(!initialized_);
  if (initialized_) {
    return Status::IllegalState("already initialized");
  }
  vector<HostPort> hostports;
  RETURN_NOT_OK(server_->GetMasterHostPorts(&hostports));
  vector<string> master_addrs;
  master_addrs.reserve(hostports.size());
  for (const auto& hp : hostports) {
    master_addrs.emplace_back(hp.ToString());
  }
  RETURN_NOT_OK(TxnSystemClient::Create(master_addrs, &txn_sys_client_));
  DCHECK(txn_sys_client_);
  auto s = txn_sys_client_->CreateTxnStatusTable(
      FLAGS_txn_manager_status_table_range_partition_span);
  if (!s.ok() && !s.IsAlreadyPresent()) {
    // Status::OK() is expected only on the very first call to Init() before
    // the transaction status table is created.
    return s;
  }
  RETURN_NOT_OK(txn_sys_client_->OpenTxnStatusTable());
  initialized_ = true;
  return Status::OK();
}

Status TxnManager::CheckInitialized(const MonoTime& deadline) {
  static const auto kTransientErrStatus = Status::ServiceUnavailable(
      "TxnManager is not yet initialized");

  if (initialized_) {
    return Status::OK();
  }
  if (!is_lazily_initialized_) {
    return kTransientErrStatus;
  }

  // In case of lazy initialization, calls to TxnManager trigger the
  // initialization of the object.
  bool need_init = true;
  if (need_init_.compare_exchange_strong(need_init, false)) {
    auto s = server_->ScheduleTxnManagerInit();
    // In a rare case of failure, let next call schedule the init.
    if (PREDICT_FALSE(!s.ok())) {
      LOG(WARNING) << s.ToString();
      need_init_ = true;
      return kTransientErrStatus;
    }
  }
  // TODO(aserbin): subtract a small portion of the timeout to respond a bit
  //                earlier the deadline, otherwise client might consider
  //                the call timing out, but we want to deliver
  //                ServiceUnavailable() status instead.
  auto s = server_->WaitForTxnManagerInit(
      deadline.Initialized() ? ToDelta(deadline) : MonoDelta());
  if (s.IsTimedOut()) {
    // The state of not-yet-initialized TxnManager is a transitional one,
    // so callers are assumed to retry and succeed eventually.
    return kTransientErrStatus;
  }
  return s;
}

} // namespace transactions
} // namespace kudu
