import numpy as np  # Added missing import
import nn

class LanguageClassificationModel(object):
    """
    A model for language identification at a single-word granularity.
    """
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        
        # Initialize parameters
        self.w = nn.Parameter(47, 100)
        self.w_h1 = nn.Parameter(100, 100)
        self.w_h2 = nn.Parameter(100, 100)
        self.w_f = nn.Parameter(100, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.
        """
        def f(x, h):
            if not h:
                return nn.Linear(x, self.w)
            return nn.Linear(
                nn.ReLU(
                    nn.Add(
                        nn.Linear(x, self.w),
                        nn.Linear(h, self.w_h1)
                    )
                ),
                self.w_h2
            )
        
        h = None
        for x in xs:
            h = f(x, h)
        return nn.Linear(h, self.w_f)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        acc = 0
        alpha = -0.05
        while acc < 0.86:
            for xs, y in dataset.iterate_once(100):
                loss = self.get_loss(xs, y)
                grad_w, grad_w_h1, grad_w_h2, grad_w_f = nn.gradients(
                    loss, [self.w, self.w_h1, self.w_h2, self.w_f]
                )

                self.w.update(grad_w, alpha)
                self.w_h1.update(grad_w_h1, alpha)
                self.w_h2.update(grad_w_h2, alpha)
                self.w_f.update(grad_w_f, alpha)
            
            acc = dataset.get_validation_accuracy()
            print(f"Validation accuracy: {acc:.2%} | Learning rate: {alpha}")  # Improved formatting

    def save(self, prefix='model'):
        """Saves model parameters to .npy files"""
        np.save(f'{prefix}_w.npy', self.w.data)
        np.save(f'{prefix}_w_h1.npy', self.w_h1.data)
        np.save(f'{prefix}_w_h2.npy', self.w_h2.data)
        np.save(f'{prefix}_w_f.npy', self.w_f.data)

    def load(self, prefix='model'):
        """Loads model parameters from .npy files"""
        self.w.data = np.load(f'{prefix}_w.npy')
        self.w_h1.data = np.load(f'{prefix}_w_h1.npy')
        self.w_h2.data = np.load(f'{prefix}_w_h2.npy')
        self.w_f.data = np.load(f'{prefix}_w_f.npy')
