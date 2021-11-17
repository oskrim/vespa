// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#pragma once

#include "executor_thread_service.h"
#include <vespa/searchcorespi/index/ithreadingservice.h>
#include <vespa/vespalib/util/threadstackexecutor.h>

namespace proton {

class ExecutorThreadingServiceStats;
class ThreadingServiceConfig;

/**
 * Implementation of IThreadingService using 2 underlying thread stack executors
 * with 1 thread each.
 */
class ExecutorThreadingService : public searchcorespi::index::IThreadingService
{
private:
    vespalib::ThreadExecutor                           & _sharedExecutor;
    vespalib::ThreadStackExecutor                        _masterExecutor;
    std::atomic<uint32_t>                                _master_task_limit;
    std::unique_ptr<vespalib::SyncableThreadExecutor>    _indexExecutor;
    std::unique_ptr<vespalib::SyncableThreadExecutor>    _summaryExecutor;
    ExecutorThreadService                                _masterService;
    ExecutorThreadService                                _indexService;
    ExecutorThreadService                                _summaryService;
    std::unique_ptr<vespalib::ISequencedTaskExecutor>    _indexFieldInverter;
    std::unique_ptr<vespalib::ISequencedTaskExecutor>    _indexFieldWriter;
    std::unique_ptr<vespalib::ISequencedTaskExecutor>    _attributeFieldWriter;
    std::unique_ptr<vespalib::ISequencedTaskExecutor>    _field_writer;
    vespalib::ISequencedTaskExecutor*                    _index_field_inverter_ptr;
    vespalib::ISequencedTaskExecutor*                    _index_field_writer_ptr;
    vespalib::ISequencedTaskExecutor*                    _attribute_field_writer_ptr;

    void syncOnce();
public:
    using OptimizeFor = vespalib::Executor::OptimizeFor;
    /**
     * Convenience constructor used in unit tests.
     */
    ExecutorThreadingService(vespalib::ThreadExecutor& sharedExecutor, uint32_t num_treads = 1);

    ExecutorThreadingService(vespalib::ThreadExecutor& sharedExecutor,
                             const ThreadingServiceConfig& cfg,
                             uint32_t stackSize = 128 * 1024);
    ~ExecutorThreadingService() override;

    void sync_all_executors() override;

    void blocking_master_execute(vespalib::Executor::Task::UP task) override;

    void shutdown();

    uint32_t master_task_limit() const {
        return _master_task_limit.load(std::memory_order_relaxed);
    }
    void set_task_limits(uint32_t master_task_limit,
                         uint32_t field_task_limit,
                         uint32_t summary_task_limit);

    // Expose the underlying executors for stats fetching and testing.
    // TOD: Remove - This is only used for casting to check the underlying type
    vespalib::ThreadExecutor &getMasterExecutor() {
        return _masterExecutor;
    }
    vespalib::ThreadExecutor &getIndexExecutor() {
        return *_indexExecutor;
    }
    vespalib::ThreadExecutor &getSummaryExecutor() {
        return *_summaryExecutor;
    }

    searchcorespi::index::IThreadService &master() override {
        return _masterService;
    }
    searchcorespi::index::IThreadService &index() override {
        return _indexService;
    }

    searchcorespi::index::IThreadService &summary() override {
        return _summaryService;
    }
    vespalib::ThreadExecutor &shared() override {
        return _sharedExecutor;
    }

    vespalib::ISequencedTaskExecutor &indexFieldInverter() override;
    vespalib::ISequencedTaskExecutor &indexFieldWriter() override;
    vespalib::ISequencedTaskExecutor &attributeFieldWriter() override;
    ExecutorThreadingServiceStats getStats();
};

}


