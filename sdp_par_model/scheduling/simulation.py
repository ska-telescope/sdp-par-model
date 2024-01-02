"""Functions for calculating and displaying SDP schedule simulations."""
from sdp_par_model.parameters.definitions import Telescopes, Constants, HPSOs, Pipelines
from sdp_par_model.scheduling import graph, scheduler, efficiency
from sdp_par_model.config import PipelineConfig
from sdp_par_model import reports

import multiprocessing
from matplotlib import pylab
from IPython.display import display, Markdown
from ipywidgets import interact_manual, SelectMultiple
import math
import random
import time
import sys


class ScheduleSimulation:
    """Class for simulating the schedule within a Jupyter notebook."""
    def __init__(self):
        pass

    def observatory_sizes_and_rates(self, scenario):
        # Assumptions about throughput per size for hot and cold buffer
        self.hot_rate_per_size = (
            3 * Constants.giga / 10 / Constants.tera
        )  # 3 GB/s per 10 TB [NVMe SSD]
        self.cold_rate_per_size = (
            0.5 * Constants.giga / 16 / Constants.tera
        )  # 0.5 GB/s per 16 TB [SATA SSD]

        # Common system sizing
        self.ingest_rate = 0.46 * Constants.tera  # Byte/s
        self.delivery_rate = self.lts_rate = int(100 / 8 * Constants.giga)  # Byte/s

        # Costing scenarios to assume
        if scenario == "low-cdr":
            self.telescope = Telescopes.SKA1_Low
            self.total_flops = int(13.8 * Constants.peta)  # FLOP/s
            self.input_buffer_size = int((0.5 * 46.0 - 0.6) * Constants.peta)  # Byte
            self.hot_buffer_size = int(0.5 * 46.0 * Constants.peta)  # Byte
            self.delivery_buffer_size = int(0.656 * Constants.peta)  # Byte
        elif scenario == "mid-cdr":
            self.telescope = Telescopes.SKA1_Mid
            self.total_flops = int(12.1 * Constants.peta)  # FLOP/s
            self.input_buffer_size = int((0.5 * 39.0 - 1.103) * Constants.peta)  # Byte
            self.hot_buffer_size = int(0.5 * 39.0 * Constants.peta)  # Byte
            self.delivery_buffer_size = int(0.03 * 39.0 * Constants.peta)  # Byte
        elif scenario == "low-adjusted":
            self.telescope = Telescopes.SKA1_Low
            self.total_flops = int(9.623 * Constants.peta)  # FLOP/s
            # self.input_buffer_size = int(30.0 * Constants.peta) # Byte # 1
            self.input_buffer_size = int(43.35 * Constants.peta)  # Byte
            # self.hot_buffer_size = int(17.5 * Constants.peta) # Byte # 1
            self.hot_buffer_size = int(25.5 * Constants.peta)  # Byte # 2
            # self.hot_buffer_size = int(27.472 * Constants.peta) # Byte
            self.delivery_buffer_size = int(0.656 * Constants.peta)  # Byte
        elif scenario == "mid-adjusted":
            self.telescope = Telescopes.SKA1_Mid
            self.total_flops = int(5.9 * Constants.peta)  # FLOP/s
            self.input_buffer_size = int(48.455 * Constants.peta)  # Byte
            self.hot_buffer_size = int(40.531 * Constants.peta)  # Byte
            self.delivery_buffer_size = int(1.103 * Constants.peta)  # Byte
        else:
            raise ValueError(
                f"Unknown costing scenario '{scenario}'. Must be one of ['low-cdr', 'mid-cdr', 'low-adjusted', 'mid-adjusted']"
            )

        display(f"Scenario for {self.telescope}:")
        display(
            Markdown(
                """| &nbsp; | Input | Hot | Output | All |&nbsp;|\n|-|-:|-:|-:|-:|-|
| Sizes: | {:.2f} | {:.2f} | {:.2f} | {:.2f} | PB |
| Rates: | {:.2f} | {:.2f} | {:.2f} | {:.2f} | TB/s |
        """.format(
                    self.input_buffer_size / Constants.peta,
                    self.hot_buffer_size / Constants.peta,
                    self.delivery_buffer_size / Constants.peta,
                    (
                        self.input_buffer_size
                        + self.hot_buffer_size
                        + self.delivery_buffer_size
                    )
                    / Constants.peta,
                    self.input_buffer_size * self.cold_rate_per_size / Constants.tera,
                    self.hot_buffer_size * self.hot_rate_per_size / Constants.tera,
                    self.delivery_buffer_size
                    * self.cold_rate_per_size
                    / Constants.tera,
                    (self.input_buffer_size + self.delivery_buffer_size)
                    * self.cold_rate_per_size
                    / Constants.tera
                    + self.hot_buffer_size * self.hot_rate_per_size / Constants.tera,
                )
            )
        )

    def read_hpso_csv(self, csv_file=None):
        if not csv_file:
            self.use_hpso = True
            csv_file = reports.newest_csv(reports.find_csvs())
        else:
            self.use_hpso = False
        self.csv = reports.strip_csv(reports.read_csv(csv_file))

    def computational_capacity(self):
        realtime_flops = 0
        realtime_flops_hpso = None
        
        if self.use_hpso:
            observations = []
            pipelines_in_observations = []
            names_in_observations = []
            for hpso in HPSOs.all_hpsos:
                if HPSOs.hpso_telescopes[hpso] != self.telescope:
                    continue
                
                
                observations.append(hpso)
                config_names = []
                pipelines = []
                for pipeline in HPSOs.hpso_pipelines[hpso]:
                    pipelines.append(pipeline)
                    config_names.append(PipelineConfig(hpso=hpso, pipeline=pipeline).describe())
                    
                pipelines_in_observations.append(pipelines)
                names_in_observations.append(config_names)
            
            
        else:
            #######################################################
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
            observations = reports.lookup_csv_observation_names(self.csv, self.telescope)
            pipelines_in_observations = []
            names_in_observations = []
            for observation in observations:
                #######################################################
                # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
                pipelines, names = reports.lookup_csv_config_names(self.csv, observation)
                pipelines_in_observations.append(pipelines)
                names_in_observations.append(names)
        
        for observation, pipelines, names in zip(observations, pipelines_in_observations, names_in_observations):
            # Sum FLOP rates over involved real-time pipelines
            rt_flops = 0
            for pipeline, name in zip(pipelines, names):
                flops = int(
                    math.ceil(
                        float(
                            reports.lookup_csv(
                                self.csv, name, "Total Compute Requirement"
                            )
                        )
                        * Constants.peta
                    )
                )
                if pipeline in Pipelines.realtime:
                    rt_flops += flops
            # Dominates?
            if rt_flops > realtime_flops:
                realtime_flops = rt_flops
                realtime_flops_hpso = observation

        # Show
        print("Realtime processing requirements:")
        batch_flops = self.total_flops - realtime_flops
        print(
            " {:.3f} Pflop/s real-time (from {}), {:.3f} Pflop/s left for batch".format(
                realtime_flops / Constants.peta,
                realtime_flops_hpso,
                batch_flops / Constants.peta,
            )
        )

        self.capacities = {
            graph.Resources.Observatory: 1,
            graph.Resources.BatchCompute: batch_flops,
            graph.Resources.RealtimeCompute: realtime_flops,
            graph.Resources.InputBuffer: self.input_buffer_size,
            graph.Resources.HotBuffer: self.hot_buffer_size,
            graph.Resources.OutputBuffer: self.delivery_buffer_size,
            graph.Resources.HotBufferRate: self.hot_rate_per_size
            * self.hot_buffer_size,
            graph.Resources.InputBufferRate: self.cold_rate_per_size
            * self.input_buffer_size,
            graph.Resources.OutputBufferRate: self.cold_rate_per_size
            * self.delivery_buffer_size,
            graph.Resources.IngestRate: self.ingest_rate,
            graph.Resources.DeliveryRate: self.delivery_rate,
            graph.Resources.LTSRate: self.lts_rate,
        }
        
    def computational_capacity_old(self):
        realtime_flops = 0
        realtime_flops_hpso = None
        for hpso in HPSOs.all_hpsos:
            if HPSOs.hpso_telescopes[hpso] != self.telescope:
                continue
            # Sum FLOP rates over involved real-time pipelines
            rt_flops = 0
            for pipeline in HPSOs.hpso_pipelines[hpso]:
                cfg_name = PipelineConfig(hpso=hpso, pipeline=pipeline).describe()
                flops = int(
                    math.ceil(
                        float(
                            reports.lookup_csv(
                                self.csv, cfg_name, "Total Compute Requirement"
                            )
                        )
                        * Constants.peta
                    )
                )
                if pipeline in Pipelines.realtime:
                    rt_flops += flops
            # Dominates?
            if rt_flops > realtime_flops:
                realtime_flops = rt_flops
                realtime_flops_hpso = hpso

        # Show
        print("Realtime processing requirements:")
        batch_flops = self.total_flops - realtime_flops
        print(
            " {:.3f} Pflop/s real-time (from {}), {:.3f} Pflop/s left for batch".format(
                realtime_flops / Constants.peta,
                realtime_flops_hpso,
                batch_flops / Constants.peta,
            )
        )

        self.capacities = {
            graph.Resources.Observatory: 1,
            graph.Resources.BatchCompute: batch_flops,
            graph.Resources.RealtimeCompute: realtime_flops,
            graph.Resources.InputBuffer: self.input_buffer_size,
            graph.Resources.HotBuffer: self.hot_buffer_size,
            graph.Resources.OutputBuffer: self.delivery_buffer_size,
            graph.Resources.HotBufferRate: self.hot_rate_per_size
            * self.hot_buffer_size,
            graph.Resources.InputBufferRate: self.cold_rate_per_size
            * self.input_buffer_size,
            graph.Resources.OutputBufferRate: self.cold_rate_per_size
            * self.delivery_buffer_size,
            graph.Resources.IngestRate: self.ingest_rate,
            graph.Resources.DeliveryRate: self.delivery_rate,
            graph.Resources.LTSRate: self.lts_rate,
        }


    def generate_graph(self, Tsequence, Tobs_min, batch_parallelism, display_node_info):
        self.Tobs_min = Tobs_min
        self.batch_parallelism = batch_parallelism
        self.hpso_sequence, self.Tobs_sum = graph.make_hpso_sequence(
            self.telescope, Tsequence, Tobs_min, verbose=True
        )
        print("{:.3f} d total".format(self.Tobs_sum / 3600 / 24))
        random.shuffle(self.hpso_sequence)
        t = time.time()
        self.nodes = graph.hpso_sequence_to_nodes(
            self.csv,
            self.hpso_sequence,
            self.capacities,
            Tobs_min,
            batch_parallelism=batch_parallelism,
        )
        print(
            "Multi-graph has {} nodes (generation took {:.3f}s)".format(
                len(self.nodes), time.time() - t
            )
        )
        if display_node_info:
            for node in self.nodes:
                print("{} ({}, t={} s)".format(node.name, node.hpso, node.time))
                for cost, amount in node.cost.items():
                    if cost in graph.Resources.units:
                        unit, mult = graph.Resources.units[cost]
                        print(" {}={:.2f} {}".format(cost, amount / mult, unit))
                for cost, amount in node.edge_cost.items():
                    if cost in graph.Resources.units:
                        unit, mult = graph.Resources.units[cost]
                        print(" -> {}={:.2f} {}".format(cost, amount / mult, unit))
                print()

    def sanity_check(self):
        cost_sum = {cost: 0 for cost in self.capacities.keys()}
        for task in self.nodes:
            for cost, amount in task.all_cost().items():
                assert (
                    cost in self.capacities
                ), "No {} capacity defined, required by {}!".format(cost, task.name)
                assert (
                    amount <= self.capacities[cost]
                ), "Not enough {} capacity to run {} ({:g}<{:g}!)".format(
                    cost, task.name, self.capacities[cost], amount
                )
                # Try to compute an average. Edges are the main wild-card here: We only know that they stay
                # around at least for the lifetime of the dependency *and* the longest dependent task.
                ttime = task.time
                if cost in task.edge_cost and len(task.rev_deps) > 0:
                    ttime += max([d.time for d in task.rev_deps])
                cost_sum[cost] += ttime * amount
        print("Best-case average loads:")
        for cost in graph.Resources.All:
            unit, mult = graph.Resources.units[cost]
            avg = cost_sum[cost] / self.Tobs_sum
            cap = self.capacities[cost]
            print(
                " {}:\t{:.3f} {} ({:.1f}% of {:.3f} {})".format(
                    cost, avg / mult, unit, avg / cap * 100, cap / mult, unit
                )
            )
            # Warn past 75%
            if avg > cap:
                print(
                    "Likely insufficient {} capacity!".format(cost),
                    file=sys.stderr,
                )

    def schedule_tasks(self):
        t = time.time()
        self.usage, self.task_time, self.task_edge_end_time = scheduler.schedule(
            self.nodes, self.capacities, verbose=False
        )
        print("Scheduling took {:.3f}s".format(time.time() - t))
        print(
            "Observing efficiency: {:.1f}%".format(
                self.Tobs_sum / self.usage[graph.Resources.Observatory].end() * 100
            )
        )
        trace_end = max(*self.task_edge_end_time.values())
        pylab.figure(figsize=(16, 16))
        pylab.subplots_adjust(hspace=0.5)
        for n, cost in enumerate(graph.Resources.All):
            levels = self.usage[cost]
            avg = levels.average(0, trace_end)
            unit, mult = graph.Resources.units[cost]
            pylab.subplot(len(self.usage), 1, n + 1)
            pylab.step(
                [0] + [t / 24 / 3600 for t in levels._trace.keys()] + [trace_end],
                [0] + [v / mult for v in levels._trace.values()] + [0],
                where="post",
            )
            pylab.title(
                "{}: {:.3f} {} average ({:.2f}%)".format(
                    cost, avg / mult, unit, avg / self.capacities[cost] * 100
                )
            )
            pylab.xlim((0, trace_end / 24 / 3600))
            pylab.xticks(range(int(trace_end) // 24 // 3600 + 1))
            pylab.ylim((0, self.capacities[cost] / mult * 1.01))
            pylab.ylabel(unit)
            if n + 1 < len(graph.Resources.All):
                pylab.gca().xaxis.set_ticklabels([])
        pylab.xlabel("Days")
        pylab.show()

    def update_rates(self, capacities):
        capacities[graph.Resources.HotBufferRate] = (
            self.hot_rate_per_size * capacities[graph.Resources.HotBuffer]
        )
        capacities[graph.Resources.InputBufferRate] = (
            self.cold_rate_per_size * capacities[graph.Resources.InputBuffer]
        )
        capacities[graph.Resources.OutputBufferRate] = (
            self.cold_rate_per_size * capacities[graph.Resources.OutputBuffer]
        )

    def efficiency_calculations(self):
        interesting_costs = [
            graph.Resources.BatchCompute,
            graph.Resources.InputBuffer,
            graph.Resources.HotBuffer,
            graph.Resources.OutputBuffer,
        ]
        linked_cost = {
            graph.Resources.HotBuffer: graph.Resources.HotBufferRate,
            graph.Resources.InputBuffer: graph.Resources.InputBufferRate,
            graph.Resources.OutputBuffer: graph.Resources.OutputBufferRate,
        }
        # Assumed price to add capacity
        cost_gradient = {
            graph.Resources.BatchCompute: 1850000 / Constants.peta,
            graph.Resources.RealtimeCompute: 1850000 / Constants.peta,
            graph.Resources.HotBuffer: 80000 / Constants.peta,
            graph.Resources.InputBuffer: 45000 / Constants.peta,
            graph.Resources.OutputBuffer: 45000 / Constants.peta,
        }
        # Assumed price of entire telescope to assign cost to inefficiences
        total_cost = 250 * Constants.mega

        @interact_manual(
            costs=SelectMultiple(options=graph.Resources.All, value=interesting_costs),
            percent=(1, 100, 1),
            percent_step=(1, 10, 1),
            count=(1, 100, 1),
            yaxis_range=(1, 20, 1),
            batch_parallelism=(1, 10, 1),
        )
        def test_sensitivity(
            costs=interesting_costs,
            percent=50,
            percent_step=5,
            count=multiprocessing.cpu_count(),
            batch_parallelism=self.batch_parallelism,
            yaxis_range=5,
            cost_change=False,
        ):
            # Calculate
            lengths = efficiency.determine_durations_batch(
                self.csv,
                self.hpso_sequence,
                costs,
                self.capacities,
                self.update_rates,
                percent,
                percent_step,
                count,
                Tobs_min=self.Tobs_min,
                batch_parallelism=batch_parallelism,
            )
            # Make graph
            graph_count = len(costs)
            pylab.figure(figsize=(8, graph_count * 4))
            pylab.subplots_adjust(hspace=0.4)
            for graph_ix, cost in enumerate(costs):
                pylab.subplot(graph_count, 1, graph_ix + 1)
                efficiency.plot_efficiencies(
                    pylab.gca(),
                    self.Tobs_sum,
                    cost,
                    self.capacities[cost],
                    lengths[cost],
                    linked_cost.get(cost),
                    self.update_rates,
                    cost_gradient.get(cost) if cost_change else None,
                    total_cost,
                )

    def failures(self):
        scheduler.schedule(
            self.nodes,
            self.capacities,
            self.task_time,
            self.task_edge_end_time,
            verbose=True,
        )
        new_capacities = dict(self.capacities)
        new_capacities[graph.Resources.InputBuffer] = (
            self.capacities[graph.Resources.InputBuffer] // 2
        )
        usage2, task_time2, task_edge_end_time2, failed_usage2 = scheduler.reschedule(
            self.nodes,
            new_capacities,
            5 * 24 * 3600,
            self.task_time,
            self.task_edge_end_time,
            verbose=False,
        )
        usage3, task_time3, task_edge_end_time3, failed_usage3 = scheduler.reschedule(
            self.nodes,
            self.capacities,
            8 * 24 * 3600,
            task_time2,
            task_edge_end_time2,
            verbose=False,
        )
        trace_end = max(*task_edge_end_time3.values())
        pylab.figure(figsize=(16, 16))
        pylab.subplots_adjust(hspace=0.5)
        for n, cost in enumerate(graph.Resources.All):
            levels = usage3[cost]
            avg = levels.average(0, trace_end)
            unit, mult = graph.Resources.units[cost]
            pylab.subplot(len(self.usage), 1, n + 1)
            for levels in [
                failed_usage2[cost] + failed_usage3[cost] + usage3[cost],
                failed_usage2[cost] + failed_usage3[cost],
            ]:
                pylab.step(
                    [0] + [t / 24 / 3600 for t in levels._trace.keys()] + [trace_end],
                    [0] + [v / mult for v in levels._trace.values()] + [0],
                    where="post",
                )
            pylab.title(
                "{}: {:.3f} {} average ({:.2f}%)".format(
                    cost, avg / mult, unit, avg / self.capacities[cost] * 100
                )
            )
            pylab.xlim((0, trace_end / 24 / 3600))
            pylab.xticks(range(int(trace_end) // 24 // 3600 + 1))
            pylab.ylim((0, self.capacities[cost] / mult * 1.01))
            pylab.ylabel(unit)
            if n + 1 < len(graph.Resources.All):
                pylab.gca().xaxis.set_ticklabels([])
        pylab.xlabel("Days")
        pylab.show()
