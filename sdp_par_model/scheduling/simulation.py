"""Functions for calculating and displaying SDP schedule simulations."""
from sdp_par_model.parameters.definitions import Telescopes, Constants
from IPython.display import display, Markdown


def observatory_sizes_and_rates(scenario):
    # Assumptions about throughput per size for hot and cold buffer
    hot_rate_per_size = 3 * Constants.giga / 10 / Constants.tera # 3 GB/s per 10 TB [NVMe SSD]
    cold_rate_per_size = 0.5 * Constants.giga / 16 / Constants.tera # 0.5 GB/s per 16 TB [SATA SSD]

    # Common system sizing
    ingest_rate = 0.46 * Constants.tera # Byte/s
    delivery_rate = lts_rate = int(100/8 * Constants.giga)  # Byte/s

    # Costing scenarios to assume
    if scenario == 'low-cdr':
        telescope = Telescopes.SKA1_Low
        total_flops = int(13.8 * Constants.peta) # FLOP/s
        input_buffer_size = int((0.5 * 46.0 - 0.6) * Constants.peta) # Byte
        hot_buffer_size = int(0.5 * 46.0 * Constants.peta) # Byte
        delivery_buffer_size = int(0.656 * Constants.peta) # Byte
    elif scenario == 'mid-cdr':
        telescope = Telescopes.SKA1_Mid
        total_flops = int(12.1 * Constants.peta) # FLOP/s
        input_buffer_size = int((0.5 * 39.0 - 1.103) * Constants.peta) # Byte
        hot_buffer_size = int(0.5 * 39.0 * Constants.peta) # Byte
        delivery_buffer_size = int(0.03 * 39.0 * Constants.peta) # Byte
    elif scenario == 'low-adjusted':
        telescope = Telescopes.SKA1_Low
        total_flops = int(9.623 * Constants.peta) # FLOP/s
        # input_buffer_size = int(30.0 * Constants.peta) # Byte # 1
        input_buffer_size = int(43.35 * Constants.peta) # Byte
        #hot_buffer_size = int(17.5 * Constants.peta) # Byte # 1
        hot_buffer_size = int(25.5 * Constants.peta) # Byte # 2
        #hot_buffer_size = int(27.472 * Constants.peta) # Byte
        delivery_buffer_size = int(0.656 * Constants.peta) # Byte
    elif scenario == 'mid-adjusted':
        telescope = Telescopes.SKA1_Mid
        total_flops = int(5.9 * Constants.peta) # FLOP/s
        input_buffer_size = int(48.455 * Constants.peta) # Byte
        hot_buffer_size = int(40.531 * Constants.peta) # Byte
        delivery_buffer_size = int(1.103 * Constants.peta) # Byte
    else:
        assert False, "Unknown costing scenario {}!".format(scenario)

    display("Scenario for {}:".format(telescope))
    display(Markdown("""| &nbsp; | Input | Hot | Output | All |&nbsp;|\n|-|-:|-:|-:|-:|-|
| Sizes: | {:.2f} | {:.2f} | {:.2f} | {:.2f} | PB |
| Rates: | {:.2f} | {:.2f} | {:.2f} | {:.2f} | TB/s |
""".format(
        input_buffer_size / Constants.peta, hot_buffer_size / Constants.peta, delivery_buffer_size / Constants.peta,
        (input_buffer_size+hot_buffer_size+delivery_buffer_size) / Constants.peta,
        input_buffer_size * cold_rate_per_size / Constants.tera,
        hot_buffer_size * hot_rate_per_size / Constants.tera,
        delivery_buffer_size * cold_rate_per_size / Constants.tera,
        (input_buffer_size + delivery_buffer_size) * cold_rate_per_size / Constants.tera +
        hot_buffer_size * hot_rate_per_size / Constants.tera)))
