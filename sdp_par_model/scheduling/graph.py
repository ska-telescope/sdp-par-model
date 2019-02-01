
import math
import uuid

from sdp_par_model import reports
from sdp_par_model.config import PipelineConfig
from sdp_par_model.parameters.definitions import \
    Pipelines, HPSOs, Constants as c

class Task(object):

    def __init__(self, name, result_name, time, cost, edge_cost):
        """Creates a new task to include in a task graph.

        :param name: Display name for the task
        :param result_name: Result of this task
        :param time: Time it takes to evaluate task
        :param cost: Resources required to run task
        :param edge_cost: Resources required to support outgoing
          edges. We assume these resources are required from the time
          this task starts to the point where the last dependent task
          finishes.
        """

        self.id = uuid.uuid4()
        self.name = name
        self.result_name = result_name
        self.time = time
        self.cost = cost
        self.edge_cost = edge_cost

        # Dependencies (add afterward)
        self.deps = set()
        self.rev_deps = set()

    def __repr__(self):
        return "Task({}, {}, time={}, cost={}, edge_cost={})".format(
            repr(self.name), repr(self.result_name), self.time, self.cost, self.edge_cost)

    def depend(self, other):
        self.deps.add(other)
        other.rev_deps.add(self)

    def all_cost(self):
        """ Returns all cost required to run this task. This includes edge costs
        both of this task and dependencies. """
        cost_set = set(self.cost.keys()).union(self.edge_cost.keys())
        for d in self.deps:
            cost_set = cost_set.union(d.edge_cost.keys())

        return {
            cost : sum([self.cost.get(cost,0), self.edge_cost.get(cost,0)] +
                       [d.edge_cost.get(cost,0) for d in self.deps])
            for cost in cost_set
            }

    def __hash__(self):
        return hash(self.id)

class Resources:

    # Pseudo-capacities
    Observatory = "observatory"

    # Compute capacities
    BatchCompute = "batch-compute" # costing FLOP/s
    RealtimeCompute = "realtime-compute" # costing FLOP/s

    # Buffer capacities
    InputBuffer = "input-buffer" # Byte
    HotBuffer = "hot-buffer" # Byte
    OutputBuffer = "output-buffer" # Byte

    # Data rates
    IngestRate = "ingest-rate" # Byte/s
    ColdBufferRate = "cold-rate" # Byte/s (in + out)
    HotBufferRate = "hot-rate" # Byte/s (in + out)
    DeliveryRate = "delivery-rate" # Byte/s (in + out)
    LTSRate = "lts-rate" # Byte/s (in + out)

    All = [ BatchCompute, RealtimeCompute,
            InputBuffer, HotBuffer, OutputBuffer,
            IngestRate, ColdBufferRate, HotBufferRate, DeliveryRate, LTSRate
    ]

class Lookup:
    """ Data to extract data from CSV in format of original telescope parameters.

    TODO: Replace this hack by serialising telescope paramameters directly..."""

    Tobs = ('Observation time', 1)
    Ringest_total = ('Total buffer ingest rate', c.tera)
    Rio = ('Visibility I/O Rate', c.tera)
    Rflop = ('Total Compute Requirement', c.peta)
    Mbuf_vis = ('Visibility Buffer', c.peta)
    Mw_cache = ('Working (cache) memory', c.tera)
    Mout = ('Output Size', c.tera)
    Tpoint = ('Pointing Time', 1)
    Texp = ('Total Time', 1)
    Mimage_cube = ('Image cube size', c.giga)
    Mcal_out = ('Calibration output', c.giga)

def lookup_csv(csv, cfg_name, lookup):
    """ Lookup (resource) value from CSV
    :param csv: CSV table as returned by reports.read_csv
    :param cfg_name: Configuration name to read values for
    :param resource: Resource name and multiplier
    """

    name, multiplier = lookup
    result = float(reports.lookup_csv(csv, cfg_name, name))
    if result is None:
        return None
    return int(math.ceil(multiplier * result))

def make_receive_rt(csv, cfg):
    """ Generate task(s) for Receive + RT processing.

    Last node returned will have measurement data in cold buffer
    """

    Tobs = None
    Rflop = 0
    pips = []
    for pipeline in HPSOs.hpso_pipelines[cfg['hpso']]:
        if pipeline not in Pipelines.realtime:
            continue
        pips.append(pipeline)
        cfg_name = PipelineConfig(pipeline=pipeline, **cfg).describe()

        # Get parameters
        if pipeline == Pipelines.Ingest:
            Tobs = lookup_csv(csv, cfg_name, Lookup.Tobs)
            Ringest = lookup_csv(csv, cfg_name, Lookup.Ringest_total)
            Mbuf_vis = lookup_csv(csv, cfg_name, Lookup.Mbuf_vis)
        Rflop += lookup_csv(csv, cfg_name, Lookup.Rflop)
    assert Tobs is not None, "No ingest pipeline for HPSO {}?".format(cfg['hpso'])

    receive_rt = Task(
        " + ".join(pips), "Measurements", Tobs,
        { Resources.Observatory: 1,
          Resources.RealtimeCompute: Rflop,
          Resources.ColdBufferRate: Ringest,
          Resources.IngestRate: Ringest},
        { Resources.InputBuffer: Mbuf_vis
            # TODO: RCAL calibration outputs?
        }
    )
    return [receive_rt]

def make_stage_buffer(inp, transfer_rate):
    """ Generate task(s) for staging data to hot buffer

    Last node returned will have data in hot buffer
    """

    stage_size = inp.edge_cost[Resources.InputBuffer]
    stage = Task(
        "Stage {}".format(inp.result_name), inp.result_name,
        stage_size / transfer_rate,
        { Resources.ColdBufferRate: transfer_rate,
          Resources.HotBufferRate: transfer_rate }, # Some compute for staging resources?
        { Resources.HotBuffer: stage_size }
    )
    stage.depend(inp)
    return [stage]

def make_offline(csv, cfg, inp, flop_rate, hot_buffer_rate):
    """Generate task(s) for off-line processing. Task results will be
    data products.

    :param csv: Performance oracle
    :param cfg: Pipeline configuration to generate tasks for
    :param inp: Input task (measurements in hot buffer)
    :param max_flop_rate: Maximum computational capacity available
    """

    assert cfg['hpso'] is not None

    # Collect offline configs
    cfg_time = {}
    cfg_floprate = {}
    cfg_iorate = {}
    cfg_output = {}
    for pipeline in HPSOs.hpso_pipelines[cfg['hpso']]:
        if pipeline in Pipelines.realtime:
            continue
        cfg_name = PipelineConfig(pipeline=pipeline, **cfg).describe()

        # Get parameters
        Tobs = lookup_csv(csv, cfg_name, Lookup.Tobs)
        Rflop = lookup_csv(csv, cfg_name, Lookup.Rflop)
        Rio = lookup_csv(csv, cfg_name, Lookup.Rio)
        cfg_output[pipeline] = lookup_csv(csv, cfg_name, Lookup.Mout)

        # We assume that DPrepD does not actually need to read
        # visibilities itself as long as it is not the only non-ICAL
        # pipeline. The idea is that it is cheap enough that we could
        # just have any other pipeline emit averaged visibilities as a
        # side-effect, skipping re-reading the visibility set.
        if pipeline == Pipelines.DPrepD:
            Rio = 0

        # Scale from observation time (Rflop) to computation time
        # (flop_rate).
        if Rflop / flop_rate > Rio / hot_buffer_rate:
            cfg_floprate[pipeline] = flop_rate
            cfg_time[pipeline] = int(Tobs * Rflop / flop_rate)
            cfg_iorate[pipeline] = int(Rio / Rflop * flop_rate)
        else:
            cfg_iorate[pipeline] = hot_buffer_rate
            cfg_time[pipeline] = int(Tobs * Rio / hot_buffer_rate)
            cfg_floprate[pipeline] = int(Rflop / Rio * hot_buffer_rate)

    offline = Task(
        " + ".join(cfg_time.keys()), "Data Products",
        sum(cfg_time.values()),
        {
            Resources.BatchCompute: max(*cfg_floprate.values()),
            Resources.HotBufferRate: max(*cfg_iorate.values()),
        }, # Some compute for staging resources?
        {
            Resources.OutputBuffer: sum(cfg_output.values())
        }
    )
    offline.depend(inp)
    return [offline]

def make_deliver(csv, inp, delivery_rate):

    deliver_size = inp.edge_cost[Resources.OutputBuffer]
    deliver = Task(
        "Deliver {}".format(inp.result_name), inp.result_name,
        deliver_size / delivery_rate,
        { Resources.ColdBufferRate: delivery_rate,
          Resources.DeliveryRate: delivery_rate },
        { }
    )
    deliver.depend(inp)
    return [deliver]

def make_graph(csv, cfg,
               cold_transfer_rate, offline_flop_rate, hot_buffer_rate, delivery_rate):
    """ Generates complete graph of tasks for an HPSO. """

    # Do receive + realtime processing. If all pipelines associated
    # with the HPSO are real-time, this is all we need to do
    receive_rt = make_receive_rt(csv, cfg)
    if all([ pip in Pipelines.realtime for pip in HPSOs.hpso_pipelines[cfg['hpso']]]):
        return receive_rt

    # Stage measurements to hot buffer
    stage = make_stage_buffer(receive_rt[-1], cold_transfer_rate)

    # Do offline processing
    offline = make_offline(csv, cfg, stage[-1], offline_flop_rate, hot_buffer_rate)

    # Delivery
    deliver = make_deliver(csv, offline[-1], delivery_rate)

    return receive_rt + stage + offline + deliver