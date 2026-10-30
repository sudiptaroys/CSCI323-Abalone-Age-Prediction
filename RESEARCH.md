# Research and conceptual basis

The problem is supervised **regression**, with eight original predictors and age = Rings + 1.5 years. Sex is encoded as F/I/M indicators, giving ten model inputs. The four distinct designs are a linear model, bagged trees, boosted trees and a feed-forward neural network. A tuned random forest is a variant of the forest method, not a fifth method.

## Related application paper

S. Guney, I. Kilinc, A. A. Hameed and A. Jamil (2022), *Abalone Age Prediction Using Machine Learning*, CCIS 1543, pp. 329–338. [DOI: 10.1007/978-3-031-04112-9_25](https://doi.org/10.1007/978-3-031-04112-9_25).

The paper studies abalone age prediction and includes machine-learning approaches relevant to the project. Its reported task uses **three age classes**. This repository predicts a continuous age; its MAE/MSE/R² cannot be compared numerically with that paper's classification accuracy. It provides application motivation, not an exact benchmark replication.

## Methods and interpretation

- **Linear regression:** minimizes sum of squared residuals. A useful interpretable baseline, but a linear decision surface cannot directly express arbitrary interactions. With the retained full dummy encoding, coefficients are non-unique; fitted predictions remain well defined through the least-squares pseudoinverse.
- **Random Forest:** averages bootstrapped randomized trees. Decorrelation and averaging can reduce variance, but predictions extrapolate poorly outside the observed target range. Training-only cross-validation controls parameter selection.
- **XGBoost:** adds regularized trees to its own ensemble, using gradient and curvature information. Shrinkage, depth restrictions and subsampling manage complexity. It does not learn from errors made by the separate random forest.
- **MLP:** composes affine transformations and ReLU nonlinearities, with squared loss and Adam updates. Backpropagation applies the chain rule through the computation graph. Scaling reduces uneven numerical conditioning; capacity and iterative optimization introduce overfitting and convergence risks.

The supervised-learning framework (hypothesis, loss and optimization) and computation-graph chain rule connect to the course's neural-network lecture. Private lecture slides are not redistributed here.

## References

1. Nash et al. (1994), *Abalone*, UCI Machine Learning Repository. [DOI: 10.24432/C55C7W](https://doi.org/10.24432/C55C7W).
2. Guney et al. (2022), cited above.
3. Breiman (2001), *Random Forests*, Machine Learning 45, 5–32. [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
4. Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*, KDD, 785–794. [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
5. Rumelhart, Hinton and Williams (1986), *Learning representations by back-propagating errors*, Nature 323, 533–536. [DOI: 10.1038/323533a0](https://doi.org/10.1038/323533a0).
6. Kingma and Ba (2015), *Adam: A Method for Stochastic Optimization*, ICLR. [Paper](https://arxiv.org/abs/1412.6980).
7. Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python*, JMLR 12, 2825–2830. [Paper](https://jmlr.org/papers/v12/pedregosa11a.html).
8. Scikit-learn, [MLPRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).
9. Scikit-learn, [RandomizedSearchCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
10. XGBoost, [Python API documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
