'''
Sagemaker expects the training script to be in the root of the repository.
This script just wraps the 'main' function of whatever training script you choose.
'''

# # from training.needs.train_bio import main
# from training.needs.train_span import main
from training.relations.relation_extraction import main

if __name__ == "__main__":
    main()
