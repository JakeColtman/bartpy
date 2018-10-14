
Model
=====

The default API for a BartPy model follows the common sklearn API. In particular, it implements:

 - fit
 - predict
 - score

For example, if we just want to train the model using default parameters, we can do:

.. code-block:: python

  from bartpy.sklearnmodel import SklearnModel
  model = SklearnModel
  model.fit(X_train, y_train)
  prediction = model.predict(y_test)

The default parameters are designed to be suitable for a wide range of data, but there are a number of parameters that can be passed into the model
These parameters can be cross_validated and optimized through grid search in the normal sklearn way


.. automodule:: bartpy.sklearnmodel
   :members:


