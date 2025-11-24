# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from matplotlib.pyplot import get_cmap
import numpy as np
import torch


class Colormap(torch.nn.Module):
    def __init__(self, name):
        self.name = name
        self._cmap = get_cmap(name)
        self._lut = torch.from_numpy(self._cmap(np.arange(self._cmap.N)))

    def forward(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        X : float or int, `~torch.Tensor` or scalar
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise a tensor of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        device = X.device
        xa = X.float().cpu().clone()
        xa *= self._cmap.N
        xa[xa >= self._cmap.N] = self._cmap.N - 1
        xa[xa < 0] = 0
        xa[torch.isnan(xa)] = self._cmap.N - 1
        xa = xa.to(torch.int64)

        lut = self._lut
        rgba = lut[xa]

        return rgba.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


if __name__ == "__main__":
    colormap = Colormap("viridis")

    x = torch.linspace(0, 1, 10000)
    colors = colormap(x)
    print(colors.shape)
