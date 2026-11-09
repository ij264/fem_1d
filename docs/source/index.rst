fem_1d
======

One-dimensional finite and spectral elements for coupled, multi-field weak
forms, with adjoint solves and sensitivity kernels.

A problem is fields on a mesh (:class:`~fem_1d.FunctionSpace`), a weak form
(:class:`~fem_1d.Term`, :class:`~fem_1d.LoadTerm`, :class:`~fem_1d.PointTerm`,
:class:`~fem_1d.PointLoad`) and Dirichlet conditions (:class:`~fem_1d.Dirichlet`),
assembled and solved by :class:`~fem_1d.Problem`, which also gives adjoint
solves and sensitivity kernels.

.. toctree::
   :maxdepth: 2

   api
