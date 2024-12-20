import numpy as np

import syntax_shap.syntaxshap.links as links
from syntax_shap.syntaxshap.models import Model
from syntax_shap.syntaxshap.utils import MaskedModel

from syntax_shap.syntaxshap.explainers._explainer import Explainer


class Random(Explainer):
    """ Simply returns random (normally distributed) feature attributions.

    This is only for benchmark comparisons. It supports both fully random attributions and random
    attributions that are constant across all explanations.
    """
    def __init__(self, model, masker, link=links.identity, feature_names=None, linearize_link=True, constant=False, **call_args):
        super().__init__(model, masker, link=link, linearize_link=linearize_link, feature_names=feature_names)

        if not isinstance(model, Model):
            self.model = Model(model)

        for arg in call_args:
            self.__call__.__kwdefaults__[arg] = call_args[arg]

        self.constant = constant
        self.constant_attributions = None
        print(self.model)

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """ Explains a single row.
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # compute any custom clustering for this row
        row_clustering = None
        if getattr(self.masker, "clustering", None) is not None:
            if isinstance(self.masker.clustering, np.ndarray):
                row_clustering = self.masker.clustering
            elif callable(self.masker.clustering):
                row_clustering = self.masker.clustering(*row_args)
            else:
                raise NotImplementedError("The masker passed has a .clustering attribute that is not yet supported by the Permutation explainer!")

        # compute the correct expected value
        masks = np.zeros(1, dtype=int)
        outputs = fm(masks, zero_index=0, batch_size=1)
        expected_value = outputs[0]

        # generate random feature attributions
        # we produce small values so our explanation errors are similar to a constant function
        row_values = np.random.randn(*((len(fm),) + outputs.shape[1:])) * 0.001

        mask_shapes = []
        for s in fm.mask_shapes:
            s = list(s)
            s[0] -= self.keep_prefix + self.keep_suffix
            mask_shapes.append(tuple(s))

        return {
            "values": row_values,
            "expected_values": expected_value,
            "mask_shapes": [(s[0],1) for s in mask_shapes],
            "main_effects": None,
            "clustering": row_clustering,
            "error_std": None,
            "output_names": self.model.output_names if hasattr(self.model, "output_names") else None
        }

