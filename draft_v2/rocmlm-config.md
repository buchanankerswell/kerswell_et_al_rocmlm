| Model   | Hyperparameter     | Value                                      | Tuned   |
|:--------|:-------------------|:-------------------------------------------|:--------|
| DT      | splitter           | (best, random)                             | tuned   |
|         | max features       | (1, 2, 3)                                  | tuned   |
|         | min samples leaf   | (1, 2, 3)                                  | tuned   |
|         | min samples split  | (2, 4, 6)                                  | tuned   |
| KN      | n neighbors        | (2, 4, 8)                                  | tuned   |
|         | weights            | (uniform, distance)                        | tuned   |
| NN1     | hidden layer sizes | (8, 16, 32)                                | tuned   |
| NN2     | hidden layer sizes | ([16, 16], [32, 16], [32, 32])             | tuned   |
| NN3     | hidden layer sizes | ([32, 16, 16], [32, 32, 16], [32, 32, 32]) | tuned   |
| NN(all) | learning rate      | (0.001, 0.005, 0.001)                      | tuned   |
|         | batch size         | 20%                                        | fixed   |
|         | max epochs         | 100                                        | fixed   |