// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "spi_bm_feed_handler.h"
#include "pending_tracker.h"
#include "bucket_info_queue.h"
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/update/documentupdate.h>
#include <vespa/metrics/loadtype.h>
#include <vespa/persistence/spi/persistenceprovider.h>

using document::Document;
using document::DocumentId;
using document::DocumentUpdate;
using storage::spi::Bucket;
using storage::spi::PartitionId;
using storage::spi::PersistenceProvider;
using storage::spi::Timestamp;

namespace feedbm {

namespace {

storage::spi::LoadType default_load_type(0, "default");
storage::spi::Context context(default_load_type, storage::spi::Priority(0), storage::spi::Trace::TraceLevel(0));

void get_bucket_info_loop(PendingTracker &tracker)
{
    auto bucket_info_queue = tracker.get_bucket_info_queue();
    if (bucket_info_queue != nullptr) {
        bucket_info_queue->get_bucket_info_loop();
    }
}

class MyOperationComplete : public storage::spi::OperationComplete
{
    std::atomic<uint32_t> &_errors;
    Bucket _bucket;
    PendingTracker& _tracker;
public:
    MyOperationComplete(std::atomic<uint32_t> &errors, const Bucket& bucket, PendingTracker& tracker);
    ~MyOperationComplete();
    void onComplete(std::unique_ptr<storage::spi::Result> result) override;
    void addResultHandler(const storage::spi::ResultHandler* resultHandler) override;
};

MyOperationComplete::MyOperationComplete(std::atomic<uint32_t> &errors, const Bucket& bucket, PendingTracker& tracker)
    : _errors(errors),
      _bucket(bucket),
      _tracker(tracker)
{
    _tracker.retain();
}

MyOperationComplete::~MyOperationComplete()
{
    _tracker.release();
}

void
MyOperationComplete::onComplete(std::unique_ptr<storage::spi::Result> result)
{
    if (result->hasError()) {
        ++_errors;
    } else {
        auto bucket_info_queue = _tracker.get_bucket_info_queue();
        if (bucket_info_queue != nullptr) {
            bucket_info_queue->put_bucket(_bucket);
        }
    }
}

void
MyOperationComplete::addResultHandler(const storage::spi::ResultHandler * resultHandler)
{
    (void) resultHandler;
}

}

SpiBmFeedHandler::SpiBmFeedHandler(PersistenceProvider& provider, bool skip_get_spi_bucket_info)
    : IBmFeedHandler(),
      _name(vespalib::string("SpiBmFeedHandler(") + (skip_get_spi_bucket_info ? "skip-get-spi-bucket-info" : "get-spi-bucket-info") + ")"),
      _provider(provider),
      _errors(0u),
      _skip_get_spi_bucket_info(skip_get_spi_bucket_info)
{
}

SpiBmFeedHandler::~SpiBmFeedHandler() = default;

void
SpiBmFeedHandler::put(const document::Bucket& bucket, std::unique_ptr<Document> document, uint64_t timestamp, PendingTracker& tracker)
{
    get_bucket_info_loop(tracker);
    Bucket spi_bucket(bucket, PartitionId(0));
    _provider.putAsync(spi_bucket, Timestamp(timestamp), std::move(document), context, std::make_unique<MyOperationComplete>(_errors, spi_bucket, tracker));
}

void
SpiBmFeedHandler::update(const document::Bucket& bucket, std::unique_ptr<DocumentUpdate> document_update, uint64_t timestamp, PendingTracker& tracker)
{
    get_bucket_info_loop(tracker);
    Bucket spi_bucket(bucket, PartitionId(0));
    _provider.updateAsync(spi_bucket, Timestamp(timestamp), std::move(document_update), context, std::make_unique<MyOperationComplete>(_errors, spi_bucket, tracker));
}

void
SpiBmFeedHandler::remove(const document::Bucket& bucket, const DocumentId& document_id,  uint64_t timestamp, PendingTracker& tracker)
{
    get_bucket_info_loop(tracker);
    Bucket spi_bucket(bucket, PartitionId(0));
    _provider.removeAsync(spi_bucket, Timestamp(timestamp), document_id, context, std::make_unique<MyOperationComplete>(_errors, spi_bucket, tracker));
}

void
SpiBmFeedHandler::create_bucket(const document::Bucket& bucket)
{
    _provider.createBucket(Bucket(bucket, PartitionId(0)), context);
}

void
SpiBmFeedHandler::attach_bucket_info_queue(PendingTracker& tracker)
{
    if (!_skip_get_spi_bucket_info) {
        tracker.attach_bucket_info_queue(_provider, _errors);
    }
}

uint32_t
SpiBmFeedHandler::get_error_count() const
{
    return _errors;
}

const vespalib::string&
SpiBmFeedHandler::get_name() const
{
    return _name;
}

bool
SpiBmFeedHandler::manages_buckets() const
{
    return false;
}

bool
SpiBmFeedHandler::manages_timestamp() const
{
    return false;
}

}
