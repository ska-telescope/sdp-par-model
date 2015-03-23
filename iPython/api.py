__author__ = 'Francois'

from parameter_definitions import Telescopes, ImagingModes, Bands
from implementation import Implementation as imp
import sympy.physics.units as u


class SKAAPI:
    """
    This class (SKA API) represents an API by which the SKA Parametric Model can be called programmatically, without
    making use of the iPython Notebook infrastructure.
    """

    def __init__(self):
        pass

    telescopes_pretty_print = {Telescopes.SKA1_Low: 'SKA1-Low',
                               Telescopes.SKA1_Mid: 'SKA1-Mid (Band 1)',
                               Telescopes.SKA1_Sur: 'SKA1-Survey (Band 1)'
                               }

    modes_pretty_print = {ImagingModes.Continuum: 'Continuum',
                          ImagingModes.Spectral: 'Spectral',
                          ImagingModes.SlowTrans: 'SlowTrans'
                          }

    @staticmethod
    def evaluate_expression(expression, tp, tsnap, nfacet):
        try:
            expression_subst = expression.subs({tp.Tsnap: tsnap, tp.Nfacet: nfacet})
            result = imp.evaluate_binned_expression(expression_subst, tp)
        except Exception as e:
            result = expression
        return result

    @staticmethod
    def evaluate_expressions(expressions, tp, tsnap, nfacet):
        """
        Evaluate a sequence of expressions by substituting the telescope_parameters into them. Returns the result
        """
        results = []
        for expression in expressions:
            result = SKAAPI.evaluate_expression(expression, tp, tsnap, nfacet)
            results.append(result)
        return results

    @staticmethod
    def compute_results(band, mode, max_baseline, nr_frequency_channels):
        """
        Computes a set of results for a given telescope in a given mode, with supplied max baseline,
        number of frequency channels.
        @param band:
        @param mode:
        @param max_baseline:
        @param nr_frequency_channels: the maximum number of frequency channels
        @return: a dictionary of result values
        """

        mode_lookup = {}
        for key in SKAAPI.modes_pretty_print:
            mode_lookup[key] = key

        # And now the results:
        tp = imp.calc_tel_params(band=band, mode=mode)  # Calculate the telescope parameters
        max_allowed_baseline = tp.baseline_bins[-1] / u.km
        if max_baseline <= max_allowed_baseline:
            tp.Bmax = max_baseline * u.km
            tp.Nf_max = nr_frequency_channels
            imp.update_derived_parameters(tp, mode=mode_lookup[mode])
            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=True)

            # The following variables will be evaluated:
            result_variable_strings = ('Mbuf_vis', 'Mw_cache',
                                       'Npix_linear', 'Rio', 'Rflop',
                                       'Rflop_grid', 'Rflop_fft', 'Rflop_proj', 'Rflop_conv', 'Rflop_phrot')


            # TODO: delete this block as it is now redundant
            # The result expressions need to be defined here as they depend on tp (updated in the line above)
            # result_expressions = (tp.Mbuf_vis / u.peta, tp.Mw_cache / u.tera, tp.Npix_linear, tp.Rio / u.tera,
            # tp.Rflop / u.peta, tp.Rflop_grid / u.peta, tp.Rflop_fft / u.peta,
            #                       tp.Rflop_proj / u.peta,
            #                       tp.Rflop_conv / u.peta, tp.Rflop_phrot / u.peta)


            #'Number of Facets', 'Snapshot Time',
            result_titles = ('Visibility Buffer', 'Working (cache) memory',
                             'Image side length', 'I/O Rate', 'Total Compute Requirement',
                             'rflop_grid', 'rflop_fft', 'rflop_proj', 'rflop_conv', 'rflop_phrot')

            # '', 'sec.',
            result_units = ('PetaBytes', 'TeraBytes', 'pixels', 'TeraBytes/s', 'PetaFLOPS',
                            'PetaFLOPS', 'PetaFLOPS', 'PetaFLOPS', 'PetaFLOPS', 'PetaFLOPS')

            assert len(result_variable_strings) == len(result_titles)
            assert len(result_variable_strings) == len(result_units)

            result_dict = {'Tsnap': tsnap, 'NFacet': nfacet}
            for variable_string in result_variable_strings:
                result_expression = eval('tp.%s' % variable_string)
                result = SKAAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)
                result_dict[variable_string] = result

            return result_dict

        else:
            raise Exception(
                'max_baseline exceeds the maximum allowed baseline of %g km for this telescope.' % max_allowed_baseline)
