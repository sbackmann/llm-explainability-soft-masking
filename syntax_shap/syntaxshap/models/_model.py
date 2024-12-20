import numpy as np
from syntax_shap.syntaxshap._serializable import Deserializer, Serializable, Serializer
from syntax_shap.syntaxshap.utils import record_import_error, safe_isinstance

try:
    import torch  # noqa: F401
except ImportError as e:
    record_import_error("torch", "torch could not be imported!", e)


class Model(Serializable):
    """ This is the superclass of all models.
    """

    def __init__(self, model=None):
        """ Wrap a callable model as a SHAP Model object.
        """
        if isinstance(model, Model):
            self.inner_model = model.inner_model
        else:
            self.inner_model = model

        if hasattr(model, "output_names"):
            self.output_names = model.output_names


    def __call__(self, *args):
        out = self.inner_model(*args)
        is_tensor = safe_isinstance(out, "torch.Tensor")
        out = out.cpu().detach().numpy() if is_tensor else np.array(out)
        return out

    def get_inputs(self, *args, padding_side='right'):
        pass

    def get_outputs(self, *args):
        """ Get the model outputs for the given inputs.
        """
        pass

    def save(self, out_file):
        """ Save the model to the given file stream.
        """
        super().save(out_file)
        with Serializer(out_file, "shap2.Model", version=0) as s:
            s.save("model", self.inner_model)

    @classmethod
    def load(cls, in_file, instantiate=True):
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap2.Model", min_version=0, max_version=0) as s:
            kwargs["model"] = s.load("model")
        return kwargs
