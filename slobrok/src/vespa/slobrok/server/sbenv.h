// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#pragma once

#include "named_service.h"
#include "rpc_server_map.h"
#include "rpc_server_manager.h"
#include "remote_slobrok.h"
#include "exchange_manager.h"
#include "configshim.h"
#include "ok_state.h"
#include "local_rpc_monitor_map.h"
#include "metrics_producer.h"
#include <vespa/config-slobroks.h>
#include <vespa/slobrok/cfg.h>
#include <vespa/vespalib/net/simple_health_producer.h>
#include <vespa/vespalib/net/simple_component_config_producer.h>

class FastOS_ThreadPool;
class FNET_Transport;
class FNET_Scheduler;
class FRT_Supervisor;

namespace slobrok {

class NamedService;
class ManagedRpcServer;
class RPCHooks;
class RemoteCheck;

/**
 * @class SBEnv
 * @brief Environmental class containing an entire server location broker
 *
 * XXX more description needed
 **/
class SBEnv : public Configurable
{
private:
    std::unique_ptr<FNET_Transport>    _transport;
    std::unique_ptr<FRT_Supervisor>    _supervisor;

    ConfigShim         _configShim;
    Configurator::UP   _configurator;
    bool               _shuttingDown;

    SBEnv(const SBEnv &);            // Not used
    SBEnv &operator=(const SBEnv &); // Not used

    void setup(const std::vector<std::string> &cfg) override;

    std::vector<std::string>                   _partnerList;
    std::string                                _me;
    RPCHooks                                   _rpcHooks;
    std::unique_ptr<RemoteCheck>               _remotechecktask;
    vespalib::SimpleHealthProducer             _health;
    MetricsProducer                            _metrics;
    vespalib::SimpleComponentConfigProducer    _components;
    LocalRpcMonitorMap                         _localRpcMonitorMap;
    ServiceMapHistory                          _globalVisibleHistory;

    void cleanup();
public:
    explicit SBEnv(const ConfigShim &shim);
    ~SBEnv();

    FNET_Transport *getTransport() { return _transport.get(); }
    FNET_Scheduler *getScheduler();
    FRT_Supervisor *getSupervisor() { return _supervisor.get(); }

    void shutdown();
    void suspend();
    void resume();

    RpcServerManager         _rpcsrvmanager;
    ExchangeManager          _exchanger;
    RpcServerMap             _rpcsrvmap;

    ServiceMapHistory& globalHistory() {
        return _globalVisibleHistory;
    }

    ServiceMapHistory& localHistory() {
        return _localRpcMonitorMap.history();
    }

    const std::string & mySpec() const { return _me; }

    bool isSuspended() const { return false; }
    bool isShuttingDown() const { return _shuttingDown; }

    int MainLoop();

    OkState addPeer(const std::string& name, const std::string &spec);
    OkState removePeer(const std::string& name, const std::string &spec);

    void countFailedHeartbeat() { _rpcHooks.countFailedHeartbeat(); }
};

} // namespace slobrok

