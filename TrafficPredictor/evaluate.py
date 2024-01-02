import argparse
import numpy as np

import matplotlib.pyplot as plt

def evaluate_model(train_loss_path, validation_loss_path, result_metrics_path):
    with open(train_loss_path, 'rb') as f:
        train_loss = np.load(f)

    with open(validation_loss_path, 'rb') as f:
        validation_loss = np.load(f)

    with open(result_metrics_path, 'rb') as f:
        result_metrics = np.load(f, allow_pickle = True)
    d = dict(enumerate(result_metrics.flatten()))
    print(d)

    plt.plot(train_loss, color = 'blue', label = 'training loss')
    plt.plot(validation_loss, color = 'orange', label = 'validation loss')
    plt.title('Training and validation losses')
    plt.legend()
    plt.show()



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--train_loss'
    )
    argparser.add_argument(
        '--validation_loss'
    )    
    argparser.add_argument(
        '--result_metrics'
    )

    args = argparser.parse_args()
    
    evaluate_model(args.train_loss, args.validation_loss, args.result_metrics)

if __name__ == '__main__':
    main()