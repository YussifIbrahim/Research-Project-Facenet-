# Triple loss using arrythmia dataset

This code contains methods for loading the arythmia dataset. All the dataset have been grouped by classes in an array. The model requires input triplets which are Anchor,positive and negative pairs. To help the model learn the ablility to separte inputs, hard triplets are selected within the selected triplets for training. This is done by get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train") function. These hard triplets could be selected online during training but i selected them offline before training and this was done by get_dataset(draw_batch_size,hard_batchs_size,norm_batchs_size,network,steps_per_epoch) function.

The network is trained for 40 epochs and it converges at the 30th epoch to obtain a model that creates embeddings which maintain similarities between similar examples. 

The model created is used to test the similarity between examples from the test set. This requires two inputs and their corresponding classes. This is obtained by form_test_data(val). This inputs are fed to compute_probs(network,X,Y) function and it finds the distance between the two inputs. It is observed that the embeddings have smaller distance between similar examples and large distance between diffferent examples.
